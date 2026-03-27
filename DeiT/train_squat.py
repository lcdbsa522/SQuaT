#!/usr/bin/env python3

import argparse
import sys
import time
import yaml
import os
import logging
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn

import numpy as np

from timm.data import (
    create_dataset, create_loader, resolve_data_config
)
from timm.models import (
    create_model, safe_model_name, resume_checkpoint, load_checkpoint
)
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.deit 

from src.quantization.modules.utils import replace_module_by_qmodule_deit
from utils_squat import (
    get_student_quant_params, 
    create_adaptor,
    compute_feature_distillation_loss
)

import math

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='OFQ + SQuaT Training')

# Dataset / Model parameters
parser.add_argument('data_dir', metavar='DIR', nargs='?', default='./data',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty). Use "torch/cifar10" or "torch/cifar100" for CIFAR datasets')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation). Use "test" for CIFAR')
parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions')

# Training settings
parser.add_argument('-b', '--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay')
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR')

# Quantization parameters 
parser.add_argument('--wq-enable', action='store_true', default=False,
                    help='enable weight quantization')
parser.add_argument('--wq-mode', default='statsq', type=str,
                    help='weight quantization mode')
parser.add_argument('--wq-bitw', default=2, type=int,
                    help='weight quantization bit width')
parser.add_argument('--aq-enable', action='store_true', default=False,
                    help='enable act quantization')
parser.add_argument('--aq-mode', default='lsq', type=str,
                    help='act quantization mode')
parser.add_argument('--aq-bitw', default=2, type=int,
                    help='act quantization bit width')
parser.add_argument('--qmodules', type=str, nargs='+', default=['blocks'],
                    help='Quantized modules')
parser.add_argument('--qk-reparam', action='store_true', default=False,
                    help='use QKR (Query-Key Reparameterization)')

# SQuaT parameters
parser.add_argument('--use-squat', action='store_true', default=False,
                    help='use SQuaT feature distillation')
parser.add_argument('--teacher', default='deit_tiny_patch16_224', type=str,
                    help='Name of teacher model')
parser.add_argument('--teacher-checkpoint', default='', type=str, metavar='PATH',
                    help='Path to teacher checkpoint')
parser.add_argument('--teacher-pretrained', action='store_true', default=False,
                    help='use pretrained teacher')
parser.add_argument('--QFeatureFlag', action='store_true', default=False,
                    help='enable feature quantization for teacher')
parser.add_argument('--feature-levels', type=int, default=2,
                    help='number of feature quantization levels')
parser.add_argument('--use-adaptor', action='store_true', default=False,
                    help='use adaptor for student features')
parser.add_argument('--use-adaptor-bn', action='store_true', default=False,
                    help='use batch norm in adaptor')
parser.add_argument('--use-student-quant-params', action='store_true', default=True,
                    help='use student quantization parameters for teacher feature quantization')
parser.add_argument('--distill-loss', type=str, default='L2', choices=['L1', 'L2', 'KL_Div'],
                    help='Feature distillation loss type')
parser.add_argument('--feature-distill-loss-type', type=str, default='L2', choices=['L1', 'L2', 'KL_Div'],
                    help='Feature distillation loss type (alias for --distill-loss)')
parser.add_argument('--kd-T', type=float, default=4.0,
                    help='temperature for knowledge distillation')
parser.add_argument('--kd-alpha', type=float, default=0.0,
                    help='weight for classification loss')
parser.add_argument('--kd-beta', type=float, default=1.0,
                    help='weight for KL divergence loss (logit distillation)')
parser.add_argument('--kd-gamma', type=float, default=1.0,
                    help='weight for feature distillation loss')
parser.add_argument('--distill-token-mode', type=str, default='all_tokens',
                    choices=['cls_only', 'all_tokens'],
                    help='Token selection mode for distillation: cls_only or all_tokens')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging')
parser.add_argument('--output', default='./outputs', type=str, metavar='PATH',
                    help='path to output folder')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment')
parser.add_argument('--gpu-id', default=0, type=int, help='gpu id to use')
parser.add_argument('--model-type', type=str, default='deit', help='model type: deit')
parser.add_argument('--workers', type=int, default=None, metavar='N',
                    help='number of data loading workers (default: auto-detect, 0 for macOS, 4 for Linux)')
parser.add_argument('--prefetcher', action='store_true', default=False,
                    help='use prefetcher for data loading')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable prefetcher for data loading')

def parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    if args.qmodules is None:
        args.qmodules = []
    
    if hasattr(args, 'feature_distill_loss_type') and args.feature_distill_loss_type:
        args.distill_loss = args.feature_distill_loss_type
    
    if not hasattr(args, 'distill_loss') or args.distill_loss is None:
        args.distill_loss = 'L2'
    
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_qat_model(model, args):
    qat_model = model
    qconfigs = {}
    
    ACT_LAYER_MAPPINGS = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'prelu': nn.PReLU,
        'rprelu': 'rprelu',
        'None': 'None'
    }
    
    for m in args.qmodules:
        wcfg = {
            "mode": args.wq_mode if args.wq_enable else "Identity",
            "bit": args.wq_bitw if args.wq_bitw < 32 and args.wq_enable else "identity",
            "all_positive": False,
            "symmetric": True,
            "per_channel": True,
            "normalize_first": False,
            "learnable": False
        }
        acfg = {
            "enable": args.aq_enable if args.aq_enable else "Identity",
            "mode": args.aq_mode if args.aq_bitw < 32 and args.aq_enable else "identity",
            "bit": args.aq_bitw,
            "per_channel": True,
            "normalize_first": False,
            "learnable": True
        }
        qconfigs[m] = {
            "weight": wcfg, 
            "act": acfg, 
            "q_attn_dropout": 0,
            "act_layer": ACT_LAYER_MAPPINGS.get('gelu', nn.GELU)
        }
    
    if args.model_type == 'deit':
        distill_token_mode = getattr(args, 'distill_token_mode', 'all_tokens')
        qat_model = replace_module_by_qmodule_deit(
            model, qconfigs, 
            pretrained_initialized=args.pretrained,
            qk_reparam=args.qk_reparam, 
            qk_reparam_type=0,
            distill_token_mode=distill_token_mode
        )
    
    return qat_model


def create_teacher_model_squat(args, is_cifar=False):
    from src.deit_vision_transformer import VisionTransformer
    
    teacher_kwargs = {
        'num_classes': args.num_classes,
        'drop_rate': 0.0,
        'pretrained': args.teacher_pretrained,  
    }
    
    if is_cifar:
        teacher_kwargs['img_size'] = 224
    
    teacher = create_model(args.teacher, **teacher_kwargs)
    
    if args.teacher_checkpoint:
        _logger.info(f"Loading teacher checkpoint from {args.teacher_checkpoint}")
        checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        teacher.load_state_dict(state_dict, strict=True)
    elif args.teacher_pretrained:
        pass
    
    teacher.model_type = 'teacher'
    teacher.QFeatureFlag = args.QFeatureFlag
    teacher.feature_levels = args.feature_levels
    teacher.use_adaptor = False
    teacher.use_adaptor_bn = False
    teacher.use_student_quant_params = args.QFeatureFlag
    
    if args.QFeatureFlag:
        try:
            from src.quantization.modules.feature_quant_module import FeatureQuantizerViT
            class Args:
                def __init__(self):
                    self.feature_levels = args.feature_levels
                    self.use_student_quant_params = args.QFeatureFlag
            feature_args = Args()
            teacher.feature_quantizer = FeatureQuantizerViT(feature_args)
            _logger.info("Added Teacher FeatureQuantizer")
        except Exception as e:
            _logger.warning(f"Failed to create FeatureQuantizer: {e}")
    
    if not hasattr(teacher, '_original_forward'):
        teacher._original_forward = teacher.forward
        
        teacher.mlp_fc1_input = None
        def hook_fn(module, input):
            teacher.mlp_fc1_input = input[0]
            
        distill_token_mode = getattr(args, 'distill_token_mode', 'all_tokens')
        
        if hasattr(teacher, 'blocks') and len(teacher.blocks) > 0:
            last_block = teacher.blocks[-1]
            if hasattr(last_block, 'mlp') and hasattr(last_block.mlp, 'fc1'):
                _logger.info(f"Registered hook to Teacher's last block MLP fc1 layer (token_mode: {distill_token_mode})")
                last_block.mlp.fc1.register_forward_pre_hook(hook_fn)
            else:
                _logger.warning("Could not find MLP fc1 layer in teacher model")
        
        def forward_with_feat(x, is_feat=False, quant_params=None):
            if is_feat:
                try:
                    output = teacher._original_forward(x)

                    if isinstance(output, tuple):
                        logit = output[0]
                    else:
                        logit = output
                    
                    feat_t = teacher.mlp_fc1_input
                    
                    if feat_t is None:
                        _logger.warning("Teacher MLP fc1 feature not captured by hook! Using logit as fallback.")
                        feat_t = logit
                    
                    if feat_t.dim() == 3:
                        if distill_token_mode == 'cls_only':
                            feat_t = feat_t[:, 0, :]  
                    
                    if teacher.QFeatureFlag and hasattr(teacher, 'feature_quantizer'):
                        if feat_t.dim() == 2:  
                            feat_t = feat_t.unsqueeze(1) 
                        fd_map = teacher.feature_quantizer(feat_t, quant_params=quant_params)
                        if distill_token_mode == 'cls_only' and fd_map.shape[1] == 1:
                            fd_map = fd_map.squeeze(1) 
                    else:
                        fd_map = feat_t
                    
                    return logit, None, None, fd_map
                    
                except Exception as e:
                    _logger.error(f"Teacher forward with feature failed: {e}")
                    raise e
            
            return teacher._original_forward(x)
            
        teacher.forward = forward_with_feat
    
    return teacher


def main(local_rank, args):
    args, args_text = args
    
    if args.no_prefetcher:
        args.prefetcher = False
    elif not hasattr(args, 'prefetcher') or args.prefetcher is False:
        args.prefetcher = False 
    args.local_rank = local_rank
    
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu_id}'
    else:
        args.device = 'cpu'
    
    args.world_size = 1
    args.rank = 0
    
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            from datetime import datetime
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                f"w{args.wq_bitw}a{args.aq_bitw}",
                "squat" if args.use_squat else "ofq"
            ])
        from timm.utils import get_outdir
        output_dir = get_outdir(args.output, exp_name)
        args.output_dir = output_dir  
    
    setup_default_logging()
    if output_dir is not None:
        log_file = os.path.join(output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        _logger.addHandler(file_handler)
        _logger.info(f"Logging to file: {log_file}")
    
    if args.device == 'cpu':
        _logger.info("CUDA not available, using CPU")
    
    random_seed(args.seed, args.rank)
    
    is_cifar = False
    if args.dataset.lower() in ['torch/cifar10', 'cifar10']:
        is_cifar = True
        if args.num_classes is None:
            args.num_classes = 10
        if args.img_size is None:
            args.img_size = 224  
        if args.val_split == 'validation':
            args.val_split = 'test'
        if not hasattr(args, 'teacher_pretrained') or not args.teacher_pretrained:
            args.teacher_pretrained = True
        _logger.info("Detected CIFAR-10 dataset. Setting img_size=224, num_classes=10, teacher_pretrained=True")
    elif args.dataset.lower() in ['torch/cifar100', 'cifar100']:
        is_cifar = True
        if args.num_classes is None:
            args.num_classes = 100
        if args.img_size is None:
            args.img_size = 224  
        if args.val_split == 'validation':
            args.val_split = 'test'
        if not hasattr(args, 'teacher_pretrained') or not args.teacher_pretrained:
            args.teacher_pretrained = True
        _logger.info("Detected CIFAR-100 dataset. Setting img_size=224, num_classes=100, teacher_pretrained=True")
    
    if args.model_type == "deit":
        model_name = args.model
        if is_cifar:
            if 'patch4' in model_name or 'patch8' in model_name:
                pass
            else:
                if 'deit_tiny' in model_name:
                    model_name = 'deit_tiny_distilled_patch16_224'
                elif 'deit_small' in model_name:
                    model_name = 'deit_small_distilled_patch16_224'
                elif 'deit_base' in model_name:
                    model_name = 'deit_base_distilled_patch16_224'
        
        try:
            distill_token_mode = getattr(args, 'distill_token_mode', 'all_tokens')
            model_kwargs = {
                'num_classes': args.num_classes,
                'drop_rate': 0.0,
                'pretrained': False if is_cifar else args.pretrained,  
                'model_type': 'student',
                'QFeatureFlag': False,
                'feature_levels': args.feature_levels,
                'use_adaptor': args.use_adaptor,
                'use_adaptor_bn': args.use_adaptor_bn,
                'use_student_quant_params': True,
                'distill_token_mode': distill_token_mode
            }

            if is_cifar:
                model_kwargs['img_size'] = 224  
            
            model = create_model(model_name, **model_kwargs)
            
            if is_cifar:
                _logger.info(f"Created student model for CIFAR: img_size=224, patch_size=16 (standard DeiT config)")
        except TypeError:
            model_kwargs = {
                'num_classes': args.num_classes,
                'drop_rate': 0.0,
                'pretrained': False if is_cifar else args.pretrained
            }
            if is_cifar:
                model_kwargs['img_size'] = 224  
            
            model = create_model(model_name, **model_kwargs)
            
            if is_cifar:
                _logger.info(f"Created student model for CIFAR: img_size=224, patch_size=16 (standard DeiT config)")
            
            _logger.warning("Model created without SQuaT parameters, adding them manually")
            
            distill_token_mode = getattr(args, 'distill_token_mode', 'all_tokens')
            model.model_type = 'student'
            model.QFeatureFlag = False
            model.feature_levels = args.feature_levels
            model.use_adaptor = args.use_adaptor
            model.use_adaptor_bn = args.use_adaptor_bn
            model.use_student_quant_params = True
            model.distill_token_mode = distill_token_mode
            
            if model.use_adaptor:
                embed_dim = model.embed_dim
                if hasattr(model, '_build_adaptor'):
                    model.adaptor = model._build_adaptor(embed_dim, model.use_adaptor_bn)
                else:
                    from utils_squat import create_adaptor
                    model.adaptor = create_adaptor(embed_dim, model.use_adaptor_bn)
                _logger.info("Added Student Adaptor")
            else:
                model.adaptor = None
            
            if not hasattr(model, '_original_forward'):
                model._original_forward = model.forward
                def forward_with_feat(x, is_feat=False, quant_params=None):
                    if is_feat:
                        try:
                            if hasattr(model, 'forward_features'):
                                import inspect
                                sig = inspect.signature(model.forward_features)
                                if 'is_feat' in sig.parameters:
                                    result = model.forward_features(x, is_feat=True, quant_params=quant_params)
                                    if isinstance(result, tuple):
                                        if len(result) >= 4:
                                            logit_input, attn, feat, fd_map = result[0], result[1], result[2], result[3]
                                        elif len(result) >= 3:
                                            logit_input, attn, feat = result[0], result[1], result[2]
                                            fd_map = None
                                        else:
                                            logit_input = result[0]
                                            attn, feat, fd_map = None, None, None
                                        
                                        if hasattr(model, 'head'):
                                            if model.dist_token is None:
                                                logit = model.head(logit_input)
                                            else:
                                                if isinstance(logit_input, tuple):
                                                    cls_logit = model.head(logit_input[0])
                                                    dist_logit = model.head_dist(logit_input[1]) if hasattr(model, 'head_dist') else None
                                                    logit = cls_logit
                                                else:
                                                    logit = model.head(logit_input)
                                            return logit, attn, feat, fd_map
                                        else:
                                            return logit_input, attn, feat, fd_map
                                    else:
                                        if hasattr(model, 'head'):
                                            logit = model.head(result)
                                            return logit, None, None, None
                                        return result, None, None, None
                        except Exception as e:
                            _logger.debug(f"forward_features with is_feat failed: {e}, trying manual extraction")
                        
                        try:
                            intermediate_features = []
                            attn_maps = []
                            
                            x_patch = model.patch_embed(x)
                            cls_token = model.cls_token.expand(x_patch.shape[0], -1, -1)
                            if model.dist_token is None:
                                x_tokens = torch.cat((cls_token, x_patch), dim=1)
                            else:
                                x_tokens = torch.cat((cls_token, model.dist_token.expand(x_patch.shape[0], -1, -1), x_patch), dim=1)
                            x_tokens = model.pos_drop(x_tokens + model.pos_embed)
                            
                            for block in model.blocks:
                                block_output = block(x_tokens)
                                if isinstance(block_output, tuple):
                                    x_tokens = block_output[0]
                                    attn = block_output[1] if len(block_output) > 1 else None
                                else:
                                    x_tokens = block_output
                                    attn = None
                                intermediate_features.append(x_tokens)
                                attn_maps.append(attn)
                            
                            x_tokens = model.norm(x_tokens)
                            
                            fd_map = None
                            if hasattr(model, 'model_type') and model.model_type == 'student' and hasattr(model, 'blocks'):
                                last_block = model.blocks[-1]
                                if hasattr(last_block, 'mlp') and hasattr(last_block.mlp, 'fc1'):
                                    fc1 = last_block.mlp.fc1
                                    if hasattr(fc1, 'saved_quantized_input') and fc1.saved_quantized_input is not None:
                                        qact = fc1.saved_quantized_input
                                        
                                        if hasattr(model, 'adaptor') and model.adaptor is not None:
                                            fd_map = model.adaptor(qact)
                                        else:
                                            fd_map = qact
                                    else:
                                        if len(intermediate_features) > 0:
                                            feat = intermediate_features[-1][:, 0, :]  
                                            if hasattr(model, 'adaptor') and model.adaptor is not None:
                                                fd_map = model.adaptor(feat)
                                            else:
                                                fd_map = feat
                            
                            if model.dist_token is None:
                                logit_input = model.pre_logits(x_tokens[:, 0])
                            else:
                                logit_input = x_tokens[:, 0]
                            
                            if hasattr(model, 'head'):
                                logit = model.head(logit_input)
                            else:
                                logit = logit_input
                            
                            return logit, attn_maps, intermediate_features, fd_map
                        except Exception as e:
                            _logger.debug(f"Manual feature extraction failed: {e}, using fallback")
                            output = model._original_forward(x)
                            if isinstance(output, tuple):
                                return output[0], None, None, None
                            return output, None, None, None
                    return model._original_forward(x)
                model.forward = forward_with_feat
    
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr'
        args.num_classes = model.num_classes
    
    model.to(args.device)
    
    teacher = None
    teacher_state_dict = None
    if args.use_squat:
        _logger.info("Creating teacher model for SQuaT")
        if is_cifar:
            original_teacher = args.teacher
            if 'patch4' in args.teacher or 'patch8' in args.teacher:
                pass  
            else:
                if 'deit_tiny' in args.teacher:
                    args.teacher = 'deit_tiny_distilled_patch16_224'
                elif 'deit_small' in args.teacher:
                    args.teacher = 'deit_small_distilled_patch16_224'
                elif 'deit_base' in args.teacher:
                    args.teacher = 'deit_base_distilled_patch16_224'
        
        teacher = create_teacher_model_squat(args, is_cifar=is_cifar)
        
        if is_cifar:
            _logger.info(f"Created teacher model for CIFAR: img_size=224, patch_size=16 (standard DeiT config, ImageNet pretrained)")
        teacher.to(args.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        
        teacher_state_dict = teacher.state_dict()
        _logger.info("Teacher model created and loaded. Will use teacher weights to initialize student.")
    
    if teacher_state_dict is not None and args.use_squat:
        _logger.info("Initializing student model from teacher weights...")
        student_state_dict = model.state_dict()
        matched_keys = []
        skipped_keys = []
        
        for key, value in teacher_state_dict.items():
            if key in student_state_dict:
                if any(skip_str in key for skip_str in ['input_quant_fn', 'output_quant_fn', 'weight_quant_fn', 
                                                         'quantizer', 's', 'init', 'adaptor']):
                    skipped_keys.append(key)
                    continue
                if student_state_dict[key].shape == value.shape:
                    student_state_dict[key].copy_(value)
                    matched_keys.append(key)
                else:
                    skipped_keys.append(key)
            else:
                skipped_keys.append(key)
        
        model.load_state_dict(student_state_dict, strict=False)
        _logger.info(f"Initialized {len(matched_keys)} parameters from teacher weights")
        if skipped_keys:
            _logger.debug(f"Skipped {len(skipped_keys)} parameters (shape mismatch or quantization-specific)")
            if len(skipped_keys) <= 10:
                _logger.debug(f"Skipped keys: {skipped_keys}")
            else:
                _logger.debug(f"Skipped keys (first 10): {skipped_keys[:10]}...")
    
    if args.initial_checkpoint:
        _logger.info(f"Loading additional checkpoint from {args.initial_checkpoint}")
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    if args.wq_enable or args.aq_enable:
        _logger.info("Applying quantization to student model...")
        model = get_qat_model(model, args)
        if hasattr(model, 'blocks'):
            last_block = model.blocks[-1]
            if hasattr(last_block, 'attn'):
                attn_type = type(last_block.attn).__name__
                _logger.info(f"Last block attention type after quantization: {attn_type}")
                if hasattr(last_block.attn, 'qkv'):
                    qkv_type = type(last_block.attn.qkv).__name__
                    _logger.info(f"Last block qkv type after quantization: {qkv_type}")
                    if hasattr(last_block.attn.qkv, 'input_quant_fn'):
                        _logger.info("✓ Quantization successfully applied to attention")
                    else:
                        _logger.warning("✗ Quantization NOT applied - qkv has no input_quant_fn")
    
    if is_cifar:
        from dataset import get_cifar10_dataloaders, get_cifar100_dataloaders
        if 'cifar10' in args.dataset.lower():
            dataset_train, dataset_eval = get_cifar10_dataloaders(
                data_folder=args.data_dir, download=True
            )
        elif 'cifar100' in args.dataset.lower():
            dataset_train, dataset_eval = get_cifar100_dataloaders(
                data_folder=args.data_dir, download=True
            )
        
        if 'cifar10' in args.dataset.lower():
            data_config = {
                'input_size': (3, 224, 224),
                'mean': (0.4914, 0.4822, 0.4465),
                'std': (0.2023, 0.1994, 0.2010),
                'interpolation': 'bilinear',
                'crop_pct': 1.0
            }
        else:  
            data_config = {
                'input_size': (3, 224, 224),
                'mean': (0.5071, 0.4867, 0.4408),
                'std': (0.2675, 0.2565, 0.2761),
                'interpolation': 'bilinear',
                'crop_pct': 1.0
            }
    else:
        data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
        
        dataset_train = create_dataset(
            args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
            batch_size=args.batch_size
        )
    
    if is_cifar:
        from torch.utils.data import DataLoader
        import platform
        is_macos = platform.system() == 'Darwin'
        num_workers = args.workers if args.workers is not None else (0 if is_macos else 4)
        pin_memory = not is_macos 
        
        loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        loader_eval = DataLoader(
            dataset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        import platform
        is_macos = platform.system() == 'Darwin'
        num_workers = args.workers if args.workers is not None else (0 if is_macos else 4)
        pin_memory = not is_macos  
        
        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=num_workers,
            distributed=False,
            pin_memory=pin_memory
        )
        
        dataset_eval = create_dataset(
            args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
            batch_size=args.batch_size
        )
        
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=num_workers,
            distributed=False,
            crop_pct=data_config['crop_pct'],
            pin_memory=pin_memory
        )
    
    if not hasattr(args, 'momentum'):
        args.momentum = 0.9
    if not hasattr(args, 'decay_rate'):
        args.decay_rate = 0.1
    if not hasattr(args, 'decay_epochs'):
        args.decay_epochs = 30
    if not hasattr(args, 'cooldown_epochs'):
        args.cooldown_epochs = 0
    if not hasattr(args, 'patience_epochs'):
        args.patience_epochs = 10
    
    _logger.info("Initializing quantizers with dummy forward pass...")
    model.train()
    with torch.no_grad():
        dummy_input = None
        for batch_idx, (input, target) in enumerate(loader_train):
            dummy_input = input.to(args.device)
            break
        if dummy_input is not None:
            try:
                _ = model(dummy_input)
                _logger.info("✓ Quantizers initialized successfully")
            except Exception as e:
                _logger.warning(f"Quantizer initialization failed: {e}, will continue anyway")
    
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    
    quantizer_params_in_optimizer = []
    quantizer_params_not_in_optimizer = []
    optimizer_param_ids = set()
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            optimizer_param_ids.add(id(param))
    
    for name, module in model.named_modules():
        try:
            if hasattr(module, 'input_quant_fn'):
                input_quant_fn = module.input_quant_fn
                if hasattr(input_quant_fn, 's'):
                    s_param = input_quant_fn.s
                    if s_param is not None and isinstance(s_param, torch.nn.Parameter):
                        if id(s_param) in optimizer_param_ids:
                            quantizer_params_in_optimizer.append(name)
                            _logger.info(f"✓ Quantizer s parameter in optimizer: {name}, requires_grad={s_param.requires_grad}, shape={s_param.shape}")
                        else:
                            quantizer_params_not_in_optimizer.append(name)
                            _logger.warning(f"✗ Quantizer s parameter NOT in optimizer: {name}, requires_grad={s_param.requires_grad}, shape={s_param.shape}")
            elif hasattr(module, 's') and isinstance(module, torch.nn.Module):
                if hasattr(module, 'learnable') and hasattr(module, 'bit'):
                    s_param = module.s
                    if s_param is not None and isinstance(s_param, torch.nn.Parameter):
                        if id(s_param) in optimizer_param_ids:
                            quantizer_params_in_optimizer.append(name)
                            _logger.info(f"✓ Quantizer s parameter in optimizer: {name}, requires_grad={s_param.requires_grad}, shape={s_param.shape}")
                        else:
                            quantizer_params_not_in_optimizer.append(name)
                            _logger.warning(f"✗ Quantizer s parameter NOT in optimizer: {name}, requires_grad={s_param.requires_grad}, shape={s_param.shape}")
        except Exception as e:
            _logger.debug(f"Error checking quantizer for {name}: {e}")
            continue
    
    if quantizer_params_not_in_optimizer:
        _logger.warning(f"Found {len(quantizer_params_not_in_optimizer)} quantizer parameters NOT in optimizer!")
        _logger.warning("Adding missing quantizer parameters to optimizer...")
        for name in quantizer_params_not_in_optimizer:
            module = dict(model.named_modules())[name]
            if hasattr(module, 'input_quant_fn') and hasattr(module.input_quant_fn, 's'):
                s_param = module.input_quant_fn.s
                if s_param is not None and s_param.requires_grad:
                    optimizer.param_groups[0]['params'].append(s_param)
                    _logger.info(f"✓ Added quantizer parameter to optimizer: {name}")
    else:
        _logger.info(f"All {len(quantizer_params_in_optimizer)} quantizer parameters are in optimizer.")
    
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    
    criterion_cls = nn.CrossEntropyLoss().to(args.device)
    criterion_div = None  
    if args.use_squat:
        from timm.loss import LabelSmoothingCrossEntropy
        criterion_div = LabelSmoothingCrossEntropy(smoothing=0.1).to(args.device)
    
    output_dir = getattr(args, 'output_dir', None)
    if output_dir is None:
        if args.rank == 0:
            if args.experiment:
                exp_name = args.experiment
            else:
                exp_name = '-'.join([
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    f"w{args.wq_bitw}a{args.aq_bitw}",
                    "squat" if args.use_squat else "ofq"
                ])
            output_dir = get_outdir(args.output, exp_name)
            args.output_dir = output_dir
    
    if output_dir is not None:
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    
    best_metric = None
    best_epoch = None
    
    for epoch in range(args.start_epoch, num_epochs):
        train_metrics = train_one_epoch(
            epoch, model, loader_train, optimizer, criterion_cls, args,
            lr_scheduler=lr_scheduler, teacher=teacher, output_dir=output_dir
        )
        
        eval_metrics = validate(
            model, loader_eval, criterion_cls, args
        )
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, eval_metrics.get('top1', 0))
        
        if output_dir is not None:
            update_summary(
                epoch, train_metrics, eval_metrics,
                os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None
            )
        
        if output_dir is not None:
            save_metric = eval_metrics.get('top1', 0)
            if best_metric is None or save_metric > best_metric:
                best_metric = save_metric
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'test_acc': save_metric,
                }, os.path.join(output_dir, 'best.pth.tar'))
    
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(epoch, model, loader, optimizer, criterion_cls, args,
                   lr_scheduler=None, teacher=None, output_dir=None):
    device = args.device
    model.train()
    if teacher is not None:
        teacher.eval()
    
    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    
    end = time.time()
    last_idx = len(loader) - 1
    
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        input = input.to(args.device)
        target = target.to(args.device)
        
        # Forward pass
        if args.use_squat and teacher is not None:
            # Student forward
            try:
                student_logit, student_attn, student_feat, fd_map_s = model(input, is_feat=True)
            except Exception as e:
                _logger.error(f"Student forward failed: {e}")
                raise
            
            quant_params = get_student_quant_params(model, feature_extraction_layer='mlp_fc1')
            if batch_idx == 0 or batch_idx % 100 == 0:
                s_info = quant_params.get('s', None)
                if s_info is not None and isinstance(s_info, torch.Tensor):
                    s_mean = s_info.mean().item() if s_info.dim() > 0 else s_info.item()
                else:
                    s_mean = s_info if s_info is not None else 'None'
                _logger.info(f"[BATCH {batch_idx}] quant_params: s={s_mean:.6f}, bit={quant_params.get('bit', 'None')}, all_positive={quant_params.get('all_positive', 'None')}")
                if hasattr(model, 'blocks') and len(model.blocks) > 0:
                    last_block = model.blocks[-1]
                    if hasattr(last_block, 'mlp') and hasattr(last_block.mlp, 'fc1'):
                        if hasattr(last_block.mlp.fc1, 'input_quant_fn'):
                            quantizer = last_block.mlp.fc1.input_quant_fn
                            if hasattr(quantizer, 's') and quantizer.s is not None:
                                s_mean = quantizer.s.mean().item() if quantizer.s.dim() > 0 else quantizer.s.item()
                                _logger.info(f"[BATCH {batch_idx}] Student quantizer s (proj, mean): {s_mean:.6f}")
            
            # Teacher forward
            with torch.no_grad():
                try:
                    teacher_logit, teacher_attn, teacher_feat, fd_map_t = teacher(input, is_feat=True, quant_params=quant_params)
                    if (batch_idx == 0 or batch_idx % 100 == 0) and hasattr(teacher, 'feature_quantizer'):
                        if hasattr(teacher.feature_quantizer, 'lsq_quantizer') and hasattr(teacher.feature_quantizer.lsq_quantizer, 's'):
                            s = teacher.feature_quantizer.lsq_quantizer.s
                            if s is not None:
                                s_mean = s.mean().item() if s.dim() > 0 else s.item()
                                bit = teacher.feature_quantizer.lsq_quantizer.bit
                                _logger.info(f"[BATCH {batch_idx}] Teacher quantizer: s_mean={s_mean:.6f}, bit={bit}, all_positive={teacher.feature_quantizer.lsq_quantizer.all_positive}")
                except Exception as e:
                    _logger.error(f"Teacher forward failed: {e}")
                    raise
            
            loss_cls = criterion_cls(student_logit, target)
            
            if batch_idx == 0 or batch_idx % args.log_interval == 0:
                _logger.info(f"[BATCH {batch_idx}] fd_map_s: {'Tensor' if fd_map_s is not None else 'None'}, fd_map_t: {'Tensor' if fd_map_t is not None else 'None'}")
                if fd_map_s is not None:
                    _logger.info(f"[BATCH {batch_idx}] fd_map_s shape: {fd_map_s.shape}")
                if fd_map_t is not None:
                    _logger.info(f"[BATCH {batch_idx}] fd_map_t shape: {fd_map_t.shape}")
            
            if fd_map_s is not None and fd_map_t is not None:
                if fd_map_s.shape != fd_map_t.shape:
                    if fd_map_s.dim() == 2 and fd_map_t.dim() == 2:
                        min_dim = min(fd_map_s.shape[1], fd_map_t.shape[1])
                        fd_map_s = fd_map_s[:, :min_dim]
                        fd_map_t = fd_map_t[:, :min_dim]
                    else:
                        fd_map_s_flat = fd_map_s.view(fd_map_s.shape[0], -1)
                        fd_map_t_flat = fd_map_t.view(fd_map_t.shape[0], -1)
                        min_dim = min(fd_map_s_flat.shape[1], fd_map_t_flat.shape[1])
                        fd_map_s = fd_map_s_flat[:, :min_dim]
                        fd_map_t = fd_map_t_flat[:, :min_dim]
                
                kd_T = getattr(args, 'kd_T', 4.0)
                try:
                    loss_fd = compute_feature_distillation_loss(fd_map_s, fd_map_t, args.distill_loss, T=kd_T)
                except Exception as e:
                    _logger.warning(f"Feature distillation loss computation failed: {e}, using 0.0")
                    loss_fd = torch.tensor(0.0).to(args.device)
            else:
                loss_fd = torch.tensor(0.0).to(args.device)
                if args.local_rank == 0 and batch_idx % args.log_interval == 0:
                    if fd_map_s is None:
                        _logger.warning("fd_map_s is None, skipping feature distillation")
                    if fd_map_t is None:
                        _logger.warning("fd_map_t is None, skipping feature distillation")
            
            loss_div = torch.tensor(0.0).to(args.device)
            if args.kd_beta > 0 and criterion_cls is not None:
                kd_T = getattr(args, 'kd_T', 4.0)
                student_log_softmax = torch.log_softmax(student_logit / kd_T, dim=1)
                teacher_softmax = torch.softmax(teacher_logit / kd_T, dim=1)
                loss_div = nn.KLDivLoss(reduction='batchmean')(
                    student_log_softmax, teacher_softmax
                ) * (kd_T ** 2)
            
            # Total loss: alpha * classification_loss + beta * logit_distill + gamma * feature_distill
            kd_alpha = getattr(args, 'kd_alpha', 0.0)
            kd_beta = getattr(args, 'kd_beta', 1.0)
            kd_gamma = getattr(args, 'kd_gamma', 1.0)
            loss = kd_alpha * loss_cls + kd_beta * loss_div + kd_gamma * loss_fd
            
            if batch_idx == 0 or batch_idx % args.log_interval == 0:
                cls_weight = kd_alpha
                _logger.info(
                    f"[BATCH {batch_idx}] Loss breakdown - "
                    f"loss_cls: {loss_cls.item():.6f} (weight={cls_weight:.2f}), "
                    f"loss_fd: {loss_fd.item():.6f}, "
                    f"loss_div: {loss_div.item():.6f}, "
                    f"total: {loss.item():.6f} "
                    f"(weights: alpha={kd_alpha:.2f}, beta={kd_beta:.2f}, gamma={kd_gamma:.2f})"
                )
            
            if not torch.isfinite(loss):
                _logger.warning(f"Non-finite loss detected: loss_cls={loss_cls}, loss_div={loss_div}, loss_fd={loss_fd}")
                if torch.isfinite(loss_fd) and loss_fd.item() > 0:
                    loss = loss_fd
                else:
                    loss = loss_cls
        else:
            output = model(input)
            student_logit = output[0] if isinstance(output, tuple) else output
            loss = criterion_cls(student_logit, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if batch_idx == 0 or batch_idx % args.log_interval == 0:
            try:
                if hasattr(model, 'blocks') and len(model.blocks) > 0:
                    last_block = model.blocks[-1]
                    if hasattr(last_block, 'attn') and hasattr(last_block.attn, 'qkv'):
                        if hasattr(last_block.attn.qkv, 'input_quant_fn'):
                            quantizer = last_block.attn.qkv.input_quant_fn
                            if hasattr(quantizer, 's') and quantizer.s is not None:
                                if quantizer.s.grad is not None:
                                    grad_norm = quantizer.s.grad.norm().item()
                                    grad_mean = quantizer.s.grad.mean().item() if quantizer.s.grad.numel() > 0 else 0.0
                                    _logger.info(f"[BATCH {batch_idx}] Quantizer s gradient: norm={grad_norm:.6f}, mean={grad_mean:.6f}")
                                else:
                                    _logger.warning(f"[BATCH {batch_idx}] Quantizer s has NO gradient!")
            except Exception as e:
                _logger.debug(f"Error checking quantizer gradient: {e}")
        
        optimizer.step()
        
        losses_m.update(loss.item(), input.size(0))
        batch_time_m.update(time.time() - end)
        
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            _logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'LR: {lr:.3e}'.format(
                    epoch, batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m, lr=lr))
        
        if lr_scheduler is not None:
            lr_scheduler.step_update(epoch * len(loader) + batch_idx)
        
        end = time.time()
    
    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, criterion, args):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.to(args.device)
            target = target.to(args.device)
            
            output = model(input)
            student_logit = output[0] if isinstance(output, tuple) else output
            
            loss = criterion(student_logit, target)
            acc1, acc5 = accuracy(student_logit, target, topk=(1, 5))
            
            losses_m.update(loss.item(), input.size(0))
            top1_m.update(acc1.item(), input.size(0))
            top5_m.update(acc5.item(), input.size(0))
            
            batch_time_m.update(time.time() - end)
            
            if last_batch or batch_idx % args.log_interval == 0:
                _logger.info(
                    'Test: [{:>4d}/{}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
            
            end = time.time()
    
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    return metrics


if __name__ == '__main__':
    args, args_text = parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    main(0, (args, args_text))

