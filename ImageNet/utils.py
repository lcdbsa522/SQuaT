from custom_modules import *
from custom_models_resnet import *
from custom_models_mobilenet import *
import numpy as  np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os, sys
from collections import OrderedDict
from tqdm import tqdm

__all__ = ['update_grad_scales']

def update_grad_scales(args, args_t, criterion_list, model_s=None, model_t=None):
    ## define model using a signle gpu
    if model_s is None:
        model_class = globals().get(args.arch)
        model_s = model_class(args)
        model_s.cuda(args.gpu)
        
        ## load weights from the last checkpoint
        checkpoint = torch.load(os.path.join(args.log_dir, 'checkpoint.pth.tar'))
        if next(iter(checkpoint['state_dict'].keys())).startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            print(model_s.load_state_dict(new_state_dict, strict=False))
        else:
            print(model_s.load_state_dict(checkpoint['state_dict'], strict=False))

    if model_t is None:
        model_class_t = globals().get(args_t.teacher_arch)
        model_t = model_class_t(args_t, pretrained=args_t.pretrained)
        model_t.cuda(args.gpu)

    ## define train lodaer in a signle gpu
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    ## define loss function
    criterion_cls, criterion_div, criterion_fd = criterion_list
    
    ## update scales
    if args.QActFlag:
        scaleA = []
    if args.QWeightFlag:
        scaleW = []
    for m in model_s.modules():
        if isinstance(m, QConv):
            m.hook_Qvalues = True
            if args.QActFlag:
                scaleA.append(0)
            if args.QWeightFlag:
                scaleW.append(0)

    model_s.train()
    # model_t.eval()
    torch.cuda.empty_cache()
    with tqdm(total=10, file=sys.stdout) as pbar:
        for num_batches, (images, labels) in enumerate(train_loader):
            if num_batches == 10: # estimate trace using 10 batches
                break
            images = images.to(args.gpu)
            labels = labels.to(args.gpu)

            # forward with single batch
            model_s.zero_grad()
            output_s, fd_map_s, quant_params = model_s(images)
            with torch.no_grad():
                output_t, fd_map_t = model_t(images, quant_params)
            
            loss_cls = criterion_cls(output_s, labels)
            loss_div = criterion_div(output_s, output_t)
            loss_fd = criterion_fd(fd_map_s, fd_map_t)
            loss_total = args.kd_alpha * loss_cls + args.kd_beta * loss_div + args.kd_gamma * loss_fd
            loss_total.backward(create_graph=True)

            # store quantized values
            if args.QWeightFlag:
                Qweight = []
            if args.QActFlag:
                Qact = []
            for m in model_s.modules():
                if isinstance(m, QConv):
                    if args.QWeightFlag:
                        Qweight.append(m.buff_weight)
                    if args.QActFlag:
                        Qact.append(m.buff_act)

            # update the scaling factor for activations
            if args.QActFlag:
                params = []
                grads = []
                for i in range(len(Qact)): # store variable & gradients
                    params.append(Qact[i])
                    grads.append(Qact[i].grad)

                for i in range(len(Qact)):
                    trace_hess_A = np.mean(trace(model_s, [params[i]], [grads[i]], 'cuda:{}'.format(args.gpu)))
                    avg_trace_hess_A = trace_hess_A / params[i].view(-1).size()[0] # avg trace of hessian
                    scaleA[i] += (avg_trace_hess_A / (grads[i].std().cpu().item()*3.0))
                
            # update the scaling factor for weights
            if args.QWeightFlag:
                params = []
                grads = []
                for i in range(len(Qweight)):
                    params.append(Qweight[i])
                    grads.append(Qweight[i].grad)

                for i in range(len(Qweight)):
                    trace_hess_W = np.mean(trace(model_s, [params[i]], [grads[i]], 'cuda:{}'.format(args.gpu)))
                    avg_trace_hess_W = trace_hess_W / params[i].view(-1).size()[0]
                    scaleW[i] += (avg_trace_hess_W / (grads[i].std().cpu().item()*3.0))
            pbar.update(1)
        
    
    if args.QActFlag:
        for i in range(len(scaleA)):
            scaleA[i] /= num_batches
            scaleA[i] = np.clip(scaleA[i], 0, np.inf)
        print("\n\nscaleA\n", scaleA)
    if args.QWeightFlag:
        for i in range(len(scaleW)):
            scaleW[i] /= num_batches
            scaleW[i] = np.clip(scaleW[i], 0, np.inf)
        print("scaleW\n", scaleW)       
    print("")

    i = 0
    for m in model_s.modules():
        if isinstance(m, QConv):
            if args.QWeightFlag:
                m.bkwd_scaling_factorW.data.fill_(scaleW[i])
            if args.QActFlag:
                m.bkwd_scaling_factorA.data.fill_(scaleA[i])
            m.hook_Qvalues = False
            i += 1

    if next(iter(checkpoint['state_dict'].keys())).startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in model_s.state_dict().items():
            name = 'module.'+ k # add `module.`
            new_state_dict[name] = v
        torch.save({'state_dict':new_state_dict},
                os.path.join(args.log_dir, 'update_scales.pth.tar'))
    else:
        torch.save({'state_dict':model_s.state_dict()},
                os.path.join(args.log_dir, 'update_scales.pth.tar'))


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv

def trace(model, params, grads, device, maxIter=50, tol=1e-3):
    """
    compute the trace of hessian using Hutchinson's method
    maxIter: maximum iterations used to compute trace
    tol: the relative tolerance
    """

    trace_vhv = []
    trace = 0.

    for i in range(maxIter):
        model.zero_grad()
        v = [
            torch.randint_like(p, high=2, device=device)
            for p in params
        ]
        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        
        Hv = hessian_vector_product(grads, params, v)
        trace_vhv.append(group_product(Hv, v).cpu().item())
        if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
            return trace_vhv
        else:
            trace = np.mean(trace_vhv)

    return trace_vhv
