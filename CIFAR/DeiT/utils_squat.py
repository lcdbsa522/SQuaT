import torch
import torch.nn as nn
import os


def load_teacher_model(model_t, teacher_path, device='cuda'):
    if os.path.isfile(teacher_path):
        print(f"Loading teacher checkpoint from '{teacher_path}'")
        checkpoint_t = torch.load(teacher_path, map_location=device)
        
        model_t.load_state_dict(checkpoint_t.get('model', checkpoint_t), strict=False)
        
        for name, p in model_t.named_parameters():
            p.requires_grad = False
        
        teacher_num_params = sum(p.numel() for p in model_t.parameters())
        teacher_num_trainable = sum(p.numel() for p in model_t.parameters() if p.requires_grad)
        
        print(f'Teacher model: {teacher_num_params} params, '
              f'{teacher_num_trainable} trainable params\n')
    else:
        raise FileNotFoundError(f"No checkpoint found at '{teacher_path}'")
    
    return model_t


def get_student_quant_params(model_s, feature_extraction_layer='mlp_fc1'):
    quant_params = {
        's': None,
        'bit': None,
        'all_positive': None
    }
    
    if hasattr(model_s, 'blocks'):
        last_block = model_s.blocks[-1]
        
        if hasattr(last_block, 'mlp'):
            mlp = last_block.mlp
            if hasattr(mlp, 'fc1') and hasattr(mlp.fc1, 'input_quant_fn'):
                quantizer = mlp.fc1.input_quant_fn
                if hasattr(quantizer, 's'):
                    if quantizer.s is not None:
                        quant_params['s'] = quantizer.s.clone().detach()
                        quant_params['bit'] = quantizer.bit
                        quant_params['all_positive'] = getattr(quantizer, 'all_positive', False)
                        return quant_params
    
    if quant_params['s'] is None:
        quant_params['s'] = torch.tensor(1.0/3.0) 
        quant_params['bit'] = 2
        quant_params['all_positive'] = False
        if not hasattr(get_student_quant_params, '_warned_defaults'):
            print("Warning: Could not extract quantization parameters, using defaults")
            get_student_quant_params._warned_defaults = True
    
    return quant_params


def create_adaptor(dim, use_bn=False, adaptor_type='linear'):
    if adaptor_type == 'linear':
        layers = [nn.Linear(dim, dim, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm1d(dim))
        return nn.Sequential(*layers)
    
    elif adaptor_type == 'mlp':
        layers = [
            nn.Linear(dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False)
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm1d(dim))
        return nn.Sequential(*layers)
    
    else:
        raise ValueError(f"Unknown adaptor type: {adaptor_type}")


def compute_feature_distillation_loss(fd_map_s, fd_map_t, loss_type='L2', T=4.0):
    if loss_type == 'L1':
        return nn.L1Loss()(fd_map_s, fd_map_t)
    elif loss_type == 'L2':
        return nn.MSELoss()(fd_map_s, fd_map_t)
    elif loss_type == 'KL_Div':
        batch_size = fd_map_s.shape[0]
        fd_s_flat = fd_map_s.view(batch_size, -1)
        fd_t_flat = fd_map_t.view(batch_size, -1)
        
        prob_s = torch.softmax(fd_s_flat / T, dim=1)
        prob_t = torch.softmax(fd_t_flat / T, dim=1)
        
        log_prob_s = torch.log_softmax(fd_s_flat / T, dim=1)
        kl_loss = nn.KLDivLoss(reduction='batchmean')(log_prob_s, prob_t)
        return kl_loss * (T ** 2) 
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

