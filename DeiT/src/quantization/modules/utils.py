import torch

from .qlinear import QLinear, QMLP
from .attention import QAttention

from src.deit_vision_transformer import Attention as deit_attention 
from src.deit_vision_transformer import Mlp

try:
    from timm.models.vision_transformer import Attention as timm_attention
    HAS_TIMM_ATTENTION = True
except ImportError:
    timm_attention = None
    HAS_TIMM_ATTENTION = False

QMODULE_MAPPINGS = {
    torch.nn.Linear: QLinear,
    deit_attention: QAttention,
    Mlp: QMLP
}

if HAS_TIMM_ATTENTION and timm_attention is not None:
    QMODULE_MAPPINGS[timm_attention] = QAttention

def get_module_by_name(model, module_name):
    names = module_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)
    return module


def set_module_by_name(model, module_name, module):
    if module_name == 'head' or module_name == 'head_dist':
        setattr(model, module_name, module)
    else:
        names = module_name.split(".")
        parent = get_module_by_name(model, ".".join(names[:-1]))
        setattr(parent, names[-1], module)


def replace_module_by_qmodule_deit(model, qconfigs, pretrained_initialized = False,
                        qk_reparam = False, qk_reparam_type = 0, boundaryRange = 0.005,
                        distill_token_mode='all_tokens'): 

        for name, cfg in qconfigs.items():
            if name == "blocks":
                if hasattr(model, 'blocks'):
                    for i, block in enumerate(model.blocks):
                            if hasattr(block, 'attn'):
                                attn_module = block.attn
                                attn_type = type(attn_module)
                                if attn_type in QMODULE_MAPPINGS:
                                    pass  
                                elif HAS_TIMM_ATTENTION and isinstance(attn_module, timm_attention) and timm_attention in QMODULE_MAPPINGS:
                                    attn_type = timm_attention 
                                elif isinstance(attn_module, deit_attention) and deit_attention in QMODULE_MAPPINGS:
                                    attn_type = deit_attention  
                                
                                if attn_type in QMODULE_MAPPINGS:
                                    device = next(attn_module.parameters()).device
                                    qmodule_attn = QMODULE_MAPPINGS[attn_type](
                                        m = attn_module,
                                        weight_bits = cfg["weight"]['bit'],
                                        input_bits = cfg["act"]['bit'],
                                        weight_channelwise = cfg["weight"]["per_channel"],
                                        input_channelwise = cfg["act"]["per_channel"],
                                        weight_quant_method = cfg["weight"]["mode"],
                                        input_quant_method = cfg["act"]["mode"],
                                        aq_learnable = cfg["act"]["learnable"],
                                        wq_learnable = cfg["weight"]["learnable"],
                                        act_layer = cfg["act_layer"],
                                        pretrained_initialized = pretrained_initialized
                                    ).to(device)  
                                    block.attn = qmodule_attn
                            
                            if hasattr(block, 'mlp'):
                                mlp_module = block.mlp
                                if type(mlp_module) in QMODULE_MAPPINGS:
                                    device = next(mlp_module.parameters()).device
                                    is_last_block = (i == len(model.blocks) - 1)
                                    is_last_mlp_fc1 = is_last_block
                                    
                                    qmodule_mlp = QMODULE_MAPPINGS[type(mlp_module)](
                                        m = mlp_module,
                                        weight_bits = cfg["weight"]['bit'],
                                        input_bits = cfg["act"]['bit'],
                                        weight_channelwise = cfg["weight"]["per_channel"],
                                        input_channelwise = cfg["act"]["per_channel"],
                                        weight_quant_method = cfg["weight"]["mode"],
                                        input_quant_method = cfg["act"]["mode"],
                                        aq_learnable = cfg["act"]["learnable"],
                                        wq_learnable = cfg["weight"]["learnable"],
                                        act_layer = cfg["act_layer"],
                                        pretrained_initialized = pretrained_initialized,
                                        is_last_mlp_fc1=is_last_mlp_fc1,
                                        distill_token_mode=distill_token_mode
                                    ).to(device)  
                                    block.mlp = qmodule_mlp
        
        if hasattr(model, 'blocks'):
            last_block = model.blocks[-1]
            if hasattr(last_block, 'mlp') and hasattr(last_block.mlp, 'fc1'):
                if hasattr(last_block.mlp.fc1, 'is_last_mlp_fc1'):
                    last_block.mlp.fc1.is_last_mlp_fc1 = True
                    print(f"Marked last MLP fc1 layer for SQuaT feature extraction (token_mode: {distill_token_mode})")

        return model
