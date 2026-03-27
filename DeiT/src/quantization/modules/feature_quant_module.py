import torch
import torch.nn as nn
from ..quantizer.lsq import LsqQuantizer


class FeatureQuantizerViT(nn.Module):
    def __init__(self, args):
        super(FeatureQuantizerViT, self).__init__()
        if hasattr(args, 'feature_levels'):
            import math
            self.feature_levels = args.feature_levels
            self.bit = int(math.log2(args.feature_levels)) if args.feature_levels > 1 else args.feature_levels
        else:
            self.feature_levels = 4  
            self.bit = 2
        
        self.use_student_quant_params = args.use_student_quant_params if hasattr(args, 'use_student_quant_params') else True
        
        self.lsq_quantizer = LsqQuantizer(
            bit=self.bit,
            all_positive=False,  
            per_channel=True,
            learnable=False 
        )
        
        self.quantizer_initialized = False
        
        self.student_s = None
        self.student_all_positive = False

    def forward(self, x, save_dict=None, quant_params=None):
        if not self.quantizer_initialized:
            with torch.no_grad():
                if quant_params is not None and 's' in quant_params and quant_params['s'] is not None:
                    student_s = quant_params['s']
                    if isinstance(student_s, torch.Tensor) and student_s.dim() > 0:
                        C = student_s.shape[0] if len(student_s.shape) == 1 else student_s.numel()
                        dummy_input = torch.zeros(1, C, device=x.device)  
                    else:
                        if len(x.shape) == 3:  
                            dummy_input = x.view(-1, x.shape[-1])[0:1]  
                        else:
                            dummy_input = x[0:1] if len(x.shape) >= 2 else x
                else:
                    if len(x.shape) == 3:  
                        dummy_input = x.view(-1, x.shape[-1])[0:1]  
                    elif len(x.shape) == 2: 
                        dummy_input = x[0:1]  
                    else:
                        dummy_input = x.view(-1, x.shape[-1])[0:1] if x.numel() > 0 else x
                
                self.lsq_quantizer.init_from(dummy_input)
                self.quantizer_initialized = True
        
        if self.use_student_quant_params and quant_params is not None:
            if 's' in quant_params and quant_params['s'] is not None:
                student_s = quant_params['s']
                if isinstance(student_s, torch.Tensor):
                    if self.lsq_quantizer.s is not None:
                        if student_s.shape == self.lsq_quantizer.s.shape:
                            self.lsq_quantizer.s.data.copy_(student_s.to(self.lsq_quantizer.s.device))
                        else:
                            if student_s.dim() == 0:
                                self.lsq_quantizer.s.data.fill_(student_s.item())
                            elif student_s.dim() == 1 and self.lsq_quantizer.s.dim() == 1:
                                if student_s.shape[0] == self.lsq_quantizer.s.shape[0]:
                                    self.lsq_quantizer.s.data.copy_(student_s.to(self.lsq_quantizer.s.device))
                                else:
                                    s_mean = student_s.mean()
                                    self.lsq_quantizer.s.data.fill_(s_mean.item())
                            else:
                                if student_s.dim() > 0:
                                    s_mean = student_s.mean()
                                else:
                                    s_mean = student_s
                                self.lsq_quantizer.s.data.fill_(s_mean.item())
                    else:
                        device = x.device
                        if student_s.dim() > 0:
                            self.lsq_quantizer.s = nn.Parameter(
                                student_s.clone().detach().to(device),
                                requires_grad=False
                            )
                        else:
                            self.lsq_quantizer.s = nn.Parameter(
                                torch.tensor(student_s.item(), device=device),
                                requires_grad=False
                            )
                    self.student_s = student_s
            
            if 'bit' in quant_params and quant_params['bit'] is not None:
                self.lsq_quantizer.bit = quant_params['bit']
            
            if 'all_positive' in quant_params and quant_params['all_positive'] is not None:
                self.lsq_quantizer.all_positive = quant_params['all_positive']
                self.student_all_positive = quant_params['all_positive']
            
            if self.lsq_quantizer.all_positive:
                if self.lsq_quantizer.bit == 1:
                    self.lsq_quantizer.thd_neg = 0
                    self.lsq_quantizer.thd_pos = 1
                else:
                    self.lsq_quantizer.thd_neg = 0
                    self.lsq_quantizer.thd_pos = 2 ** self.lsq_quantizer.bit - 1
            else:
                if self.lsq_quantizer.bit == 1:
                    self.lsq_quantizer.thd_neg = -1
                    self.lsq_quantizer.thd_pos = 1
                else:
                    self.lsq_quantizer.thd_neg = - 2 ** (self.lsq_quantizer.bit - 1)
                    self.lsq_quantizer.thd_pos = 2 ** (self.lsq_quantizer.bit - 1) - 1
        
        original_shape = x.shape
        if len(x.shape) == 3:  
            x_flat = x.view(-1, x.shape[-1])
            x_quantized = self.lsq_quantizer(x_flat)
            output = x_quantized.view(original_shape)
        else:
            output = self.lsq_quantizer(x)
        
        return output

