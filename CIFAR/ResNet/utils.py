import numpy as  np
import torch
import sys
from tqdm import tqdm
import os


from models.custom_modules import QConv
from models.feature_quant_module import FeatureQuantizer



__all__ = ['init_quant_model', 'update_grad_scales']


def printRed(skk): print("\033[91m{}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m{}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m{}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m{}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m{}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m{}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m{}\033[00m" .format(skk))


def check_trainable_parameters(model):
    trainable_num_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\033[91m'+f'Model size (trainable number of parameters): {trainable_num_para}\n'+'\033[0m')


def load_teacher_model(model_t, teacher_path):
    if os.path.isfile(teacher_path):
        print("Loading teacher checkpoint '{}'".format(teacher_path))
        checkpoint_t = torch.load(teacher_path, weights_only=True, map_location = lambda storage, loc: storage.cuda())
        model_t.load_state_dict(checkpoint_t['model'], strict=False)
        #printRed("Loaded, epoch: {}, acc: {})".format(checkpoint_t['epoch'], checkpoint_t['test_acc']))
    else:
        raise("No checkpoint found at '{}'".format(teacher_path))
    
    for name, p in model_t.named_parameters():
        p.requires_grad = False
    
    teacher_num_paramters = sum(p.numel() for p in model_t.parameters())
    teacher_num_paramters_trainable = sum(p.numel() for p in model_t.parameters() if p.requires_grad)
    print(f'Teacher model size: {teacher_num_paramters} params; Teacher trainable number of parameters: {teacher_num_paramters_trainable}\n')
    
    return model_t

### Test accuracy @ last checkpoint and best checkpoint
def test_accuracy(checkpoint_path, model, logging, device, test_loader):
    trained_model = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(trained_model['model'])
    print(f"\n{checkpoint_path} is loaded")
    logging.info(f"\n{checkpoint_path} is loaded")
    model.eval()
    with torch.no_grad():
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()

        test_acc = round(correct_classified/total*100, 4)
        print(f"Test accuracy (Top-1): {test_acc}% from epoch {trained_model['epoch']}\n")
        logging.info(f"Test accuracy (Top-1): {test_acc}% from epoch {trained_model['epoch']}\n")


def init_quant_model(model, train_loader, device, distill=None):
    for m in model.modules():
        if isinstance(m, QConv) or isinstance(m,FeatureQuantizer):
            m.init.data.fill_(1)
    
    iterloader = iter(train_loader)

    if distill == 'crd' or distill == 'crdst':
        images, labels, index, contrast_idx = next(iterloader)
    else:
        images, labels = next(iterloader) 

    images = images.to(device)

    model.train()
    model.forward(images)
    for m in model.modules():
        if isinstance(m, QConv) or isinstance(m,FeatureQuantizer):
            m.init.data.fill_(0)

def update_grad_scales(model_s, model_t, train_loader, criterion_list, device, args):
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

    if args.distill:
        criterion_cls = criterion_list[0]  # CLS Loss
        criterion_div = criterion_list[1]  # KL loss
        criterion_kd = criterion_list[2]  # MSE or MAE or KL or Cosine Loss
    else:
        criterion = criterion_list

    model_s.train()
    if args.distill:
        model_t.eval()

    with tqdm(total=3, file=sys.stdout) as pbar:
        for num_batches, data in enumerate(train_loader):
            if num_batches == 3: # estimate trace using 3 batches
                break
            
            if args.distill == "crd" or args.distill == "crdst":
                images, labels, index, contrast_idx = data
                index = index.to(device)
                contrast_idx = contrast_idx.to(device)
            else:
                images, labels = data
                index = None
                contrast_idx = None
                
            images = images.to(device)
            labels = labels.to(device)

            # forward with single batch
            model_s.zero_grad()

            if args.distill:
                flatGroupOut = True if args.distill == 'crdst' else False
                preact = False
                if args.distill in ['abound']:
                    preact = True
                
                # student model forward
                feat_s, block_out_s, logit_s, quant_params, fd_map_s = model_s(images, None, None, is_feat=True, preact=preact, flatGroupOut=flatGroupOut)
                
                # teacher model forward
                with torch.no_grad():
                    feat_t, block_out_t, logit_t, fd_map_t = model_t(images, is_feat=True, preact=preact, flatGroupOut=flatGroupOut, quant_params=quant_params)
                    feat_t = [f.detach() for f in feat_t]

                loss_cls = criterion_cls(logit_s, labels)  # CE
                loss_div = criterion_div(logit_s, logit_t)  # KL

                if args.distill == "crdst":
                    loss_kd_crd, loss_kd_crdSt = utils_distill.get_loss_crdst(args, feat_s, feat_t, criterion_kd, index, contrast_idx, block_out_s, block_out_t)
                    loss_total = args.kd_gamma * loss_cls + args.kd_alpha * loss_div + args.kd_beta * loss_kd_crd + args.kd_theta * loss_kd_crdSt
                else:
                    loss_fd = criterion_kd(fd_map_s, fd_map_t)  # L1(MAE) or L2(MSE)
                    loss_total = args.kd_alpha * loss_cls + args.kd_beta * loss_div + args.kd_gamma * loss_fd

            else:
                pred = model_s(images)
                loss_total = criterion(pred, labels)

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
                    trace_hess_A = np.mean(trace(model_s, [params[i]], [grads[i]], device))
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
                    trace_hess_W = np.mean(trace(model_s, [params[i]], [grads[i]], device))
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
    
    for param in model_s.parameters():
        param.grad = None


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