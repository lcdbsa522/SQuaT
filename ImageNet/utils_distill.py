import torch.nn as nn

from distiller_zoo import DistillKL

def define_distill_loss(args):
    criterion_cls = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_div = DistillKL(args.kd_T).cuda(args.gpu)

    if args.distill == 'fd':
        if args.distill_loss == 'L1':
            criterion_fd = nn.L1Loss().cuda(args.gpu)
        elif args.distill_loss == 'L2':
            criterion_fd = nn.MSELoss().cuda(args.gpu)
    else:
        criterion_fd = None


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_fd)

    return criterion_list