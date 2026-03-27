import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import logging
import sys
import math

from .custom_modules import QConv
from .blocks_resnet import BasicBlock, QBasicBlock
from .feature_quant_module import FeatureQuantizer



__all__ = ['resnet20_quant', 'resnet20_fp', 'resnet32_quant', 'resnet32_fp', 'resnet18_fp', 'resnet18_quant']



def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, QConv):
        nn.init.kaiming_normal_(m.weight)

class MySequential(nn.Sequential):
    def forward(self, x, save_dict=None, lambda_dict=None):
        block_num = 0
        for module in self._modules.values():
            if save_dict:
                save_dict["block_num"] = block_num
            x = module(x, save_dict, lambda_dict)
            block_num += 1
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(ResNet, self).__init__()

        self.args = args
        num_classes = args.num_classes

        print(f"\033[91mCreate ResNet, block: {block}, num_blocks: {num_blocks}, num_classes: {num_classes} \033[0m")

        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

        # student model FeatureAdaptor
        if self.args.model_type == 'student' and self.args.use_adaptor:
            self.adaptor = self.build_adaptor()
            print("Add Student Conv Adaptor layer")

        # teacher model FeatureQuantizer
        if self.args.model_type == 'teacher' and self.args.QFeatureFlag:
            self.feature_quantizer = FeatureQuantizer(args=self.args)
            print("Add Teacher FeatureQuantizer")
            
        self._mark_distill_layer()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.args, stride))
            self.in_planes = planes * block.expansion

        return MySequential(*layers)
    
    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(nn.ReLU(inplace=True)) 
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def set_replacing_rate(self, replacing_rate):
        self.args.replacing_rate = replacing_rate

    def get_student_quant_params(self, last_conv):
        quant_params = {}
        # Check for LSQ (now inside QConv)
        if hasattr(last_conv, 'sA'):
            quant_params['s'] = last_conv.sA
            quant_params['method'] = 'LSQ'
        # Check for EWGS/Standard QConv
        elif hasattr(last_conv, 'lA'):
            quant_params['lA'] = last_conv.lA.item() if hasattr(last_conv, 'lA') else None
            quant_params['uA'] = last_conv.uA.item() if hasattr(last_conv, 'uA') else None
            quant_params['method'] = 'EWGS'
        
        return quant_params
    
    def _mark_distill_layer(self):
        if self.args.model_type == 'teacher':
            last_block = self.layer3[-1]
            last_block.is_last_block = True
            print(f"Marked teacher last block in layer3: {last_block}")

        elif self.args.model_type == 'student':
            last_conv = self.layer3[-1].conv2
            if hasattr(last_conv, 'act_quantization'):
                last_conv.is_last_conv = True
                print(f"Successfully marked the last QConv layer: {last_conv}")
            else:
                print(f"Warning: Last conv layer is not a QConv instance: {type(last_conv)}")

    def build_adaptor(self):
        layers = [nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)]
        if self.args.use_adaptor_bn:
            layers.append(nn.BatchNorm2d(64))

        for m in layers:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        return nn.Sequential(*layers)

        
    def forward(self, x, save_dict=None, lambda_dict=None, is_feat=False, preact=False, flatGroupOut=False, quant_params=None):
        out = F.relu(self.bn1(self.conv1(x)))
        f0 = out

        if save_dict:   
            save_dict["layer_num"] = 1
        out = self.layer1(out, save_dict, lambda_dict)
        f1 = out

        if save_dict:
            save_dict["layer_num"] = 2
        out = self.layer2(out, save_dict, lambda_dict)
        f2 = out

        if save_dict:
            save_dict["layer_num"] = 3
        out = self.layer3(out, save_dict, lambda_dict)
        f3 = out

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        f4 = out

        out = self.bn2(out)
        out = self.linear(out)
        
        # Student Architecture
        if self.args.model_type == 'student':
            last_conv = self.layer3[-1].conv2

            if self.args.QFeatureFlag:
                quant_params = self.get_student_quant_params(last_conv)
            
            if hasattr(last_conv, 'saved_qact') and last_conv.saved_qact is not None:
                self.qact = last_conv.saved_qact

            fd_map = self.adaptor(self.qact) if self.args.use_adaptor else self.qact
        
        # Teacher Architecture
        elif self.args.model_type == 'teacher':
            last_block = self.layer3[-1]
            if hasattr(last_block, 'saved_act_conv2'):
                feat_t = last_block.saved_act_conv2
            
            fd_map = self.feature_quantizer(feat_t, quant_params=quant_params) if self.args.QFeatureFlag else feat_t


        block_out1 = [block.out for block in self.layer1]
        block_out2 = [block.out for block in self.layer2]
        block_out3 = [block.out for block in self.layer3]

        if flatGroupOut:
            f0_temp = nn.AvgPool2d(8)(f0)
            f0 = f0_temp.view(f0_temp.size(0), -1)
            f1_temp = nn.AvgPool2d(8)(f1)
            f1 = f1_temp.view(f1_temp.size(0), -1)
            f2_temp = nn.AvgPool2d(8)(f2)
            f2 = f2_temp.view(f2_temp.size(0), -1)
            f3_temp = nn.AvgPool2d(8)(f3)
            f3 = f3_temp.view(f3_temp.size(0), -1)

        if is_feat:
            if preact:
                raise NotImplementedError(f"{preact} is not implemented")
            else: 
                if self.args.model_type == 'teacher':
                    return [f0, f1, f2, f3, f4], [block_out1, block_out2, block_out3], out, fd_map
                elif self.args.model_type == 'student':
                    return [f0, f1, f2, f3, f4], [block_out1, block_out2, block_out3], out, quant_params, fd_map
                else:
                    return [f0, f1, f2, f3, f4], [block_out1, block_out2, block_out3], out
        else:
            return out


# resnet20
def resnet20_fp(args):
    return ResNet(BasicBlock, [3, 3, 3], args)

def resnet20_quant(args):
    return ResNet(QBasicBlock, [3, 3, 3], args)

# resnet32
# n = (depth - 2) // 6
def resnet32_fp(args):
    return ResNet(BasicBlock, [5, 5, 5], args)

def resnet32_quant(args):
    return ResNet(QBasicBlock, [5, 5, 5], args)

# resnet18
def resnet18_fp(args):
    return ResNet(BasicBlock, [2, 2, 2, 2], args)

def resnet18_quant(args):
    return ResNet(QBasicBlock, [2, 2, 2, 2], args)
