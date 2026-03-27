import torch
import torch.nn as nn
import logging
from .utils_quant import SymQuantizer

logger = logging.getLogger(__name__)

class FeatureQuantizerBERT(nn.Module):
    def __init__(self, args):
        super(FeatureQuantizerBERT, self).__init__()
        self.input_bits = getattr(args, 'input_bits', 8)
        self.use_student_quant_params = True
        
        self.quantizer = SymQuantizer
        
        self.clip_val = None
        
        self.squat_token = getattr(args, 'squat_token', 'cls')  # 'cls' or 'all'
        
    def forward(self, x, quant_params=None):

        if self.use_student_quant_params and quant_params is not None:
            if 'clip_val' in quant_params and quant_params['clip_val'] is not None:
                self.clip_val = quant_params['clip_val'].clone()
            if 'input_bits' in quant_params and quant_params['input_bits'] is not None:
                self.input_bits = quant_params['input_bits']
        
        if self.clip_val is None:
            self.clip_val = torch.tensor([-2.5, 2.5], device=x.device)
        
        if len(x.shape) == 3:  
            if self.squat_token == 'cls':
                x = x[:, 0, :]  
        elif len(x.shape) == 2: 
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        x_quantized = self.quantizer.apply(
            x, 
            self.clip_val, 
            self.input_bits, 
            True
        )
        
        return x_quantized

