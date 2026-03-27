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
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.deit 

import math

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Teacher Model Training (Full-precision)')

# Dataset / Model parameters
parser.add_argument('data_dir', metavar='DIR', nargs='?', default='./data',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty). Use "torch/cifar10" or "torch/cifar100" for CIFAR datasets')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation). Use "test" for CIFAR')
parser.add_argument('--model', default='deit_tiny_distilled_patch16_224', type=str, metavar='MODEL',
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
    
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


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
                "teacher"
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
        if not hasattr(args, 'pretrained') or not args.pretrained:
            args.pretrained = True
        _logger.info("Detected CIFAR-10 dataset. Setting img_size=224, num_classes=10, pretrained=True")
    elif args.dataset.lower() in ['torch/cifar100', 'cifar100']:
        is_cifar = True
        if args.num_classes is None:
            args.num_classes = 100
        if args.img_size is None:
            args.img_size = 224  
        if args.val_split == 'validation':
            args.val_split = 'test'
        if not hasattr(args, 'pretrained') or not args.pretrained:
            args.pretrained = True
        _logger.info("Detected CIFAR-100 dataset. Setting img_size=224, num_classes=100, pretrained=True")
    
    # Create teacher model 
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
        
        model_kwargs = {
            'num_classes': args.num_classes,
            'drop_rate': 0.0,
            'pretrained': args.pretrained,  
        }
        if is_cifar:
            model_kwargs['img_size'] = 224  
        
        model = create_model(model_name, **model_kwargs)
        
        if is_cifar:
            _logger.info(f"Created model with ImageNet pretrained weights for CIFAR fine-tuning: "
                        f"img_size=224, num_classes={args.num_classes}, pretrained={args.pretrained}")
    
    if args.initial_checkpoint:
        load_checkpoint(model, args.initial_checkpoint, strict=False)
    
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr'
        args.num_classes = model.num_classes
    
    model.to(args.device)
    
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
        data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
        
        dataset_train = create_dataset(
            args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
            batch_size=args.batch_size
        )
        
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
    
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    
    criterion_cls = nn.CrossEntropyLoss().to(args.device)
    
    output_dir = getattr(args, 'output_dir', None)
    if output_dir is None:
        if args.rank == 0:
            if args.experiment:
                exp_name = args.experiment
            else:
                exp_name = '-'.join([
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    "teacher"
                ])
            output_dir = get_outdir(args.output, exp_name)
            args.output_dir = output_dir
    
    if output_dir is not None:
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    
    start_epoch = args.start_epoch
    if args.resume:
        if os.path.isfile(args.resume):
            _logger.info(f'Loading checkpoint "{args.resume}"')
            checkpoint = resume_checkpoint(model, args.resume, optimizer, lr_scheduler, log_info=args.local_rank == 0)
            start_epoch = checkpoint['epoch'] + 1
            _logger.info(f'Resumed from epoch {start_epoch}')
        else:
            _logger.error(f'No checkpoint found at "{args.resume}"')
            return
    
    best_metric = None
    best_epoch = None
    
    for epoch in range(start_epoch, num_epochs):
        train_metrics = train_one_epoch(
            epoch, model, loader_train, optimizer, criterion_cls, args,
            lr_scheduler=lr_scheduler, output_dir=output_dir
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
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'test_acc': save_metric,
                }
                if lr_scheduler is not None:
                    checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
                torch.save(checkpoint, os.path.join(output_dir, 'best.pth.tar'))
                _logger.info(f'Saved best checkpoint at epoch {epoch} with accuracy {save_metric:.4f}')
    
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(epoch, model, loader, optimizer, criterion_cls, args,
                   lr_scheduler=None, output_dir=None):
    device = args.device
    model.train()
    
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
        output = model(input)
        if isinstance(output, tuple):
            logit = output[0]
            if isinstance(logit, tuple):
                logit = (logit[0] + logit[1]) / 2  
        else:
            logit = output
        loss = criterion_cls(logit, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
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
            if isinstance(output, tuple):
                logit = output[0]
                if isinstance(logit, tuple):
                    logit = (logit[0] + logit[1]) / 2 
            else:
                logit = output
            
            loss = criterion(logit, target)
            acc1, acc5 = accuracy(logit, target, topk=(1, 5))
            
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

