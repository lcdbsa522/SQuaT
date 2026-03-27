import torch
import torch.nn as nn
import math
from .custom_modules import STE_discretizer, EWGS_discretizer, FunLSQ
import logging

class FeatureQuantizer(nn.Module):
    def __init__(self, args):
        super(FeatureQuantizer, self).__init__()
        self.feature_levels = args.feature_levels
        self.baseline = args.baseline
        self.use_student_quant_params = args.use_student_quant_params
        self.quant_method = getattr(args, 'quant_method', 'EWGS') # Default to EWGS if not present
        self.distill_pos = args.distill_pos

        # EWGS Init
        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply

        self.uF = nn.Parameter(data = torch.tensor(1).float(), requires_grad = not args.use_student_quant_params)
        self.lF = nn.Parameter(data = torch.tensor(0).float(), requires_grad = not args.use_student_quant_params)
        self.register_buffer('bkwd_scaling_factorF', torch.tensor(args.bkwd_scaling_factorF).float())

        self.register_buffer('init', torch.tensor(0))
        if args.distill_pos:
            self.output_scale = nn.Parameter(torch.tensor(1.0, requires_grad=not args.use_student_quant_params))

        self.hook_Fvalues = False
        self.buff_feature = None
    
    def feature_quantization(self, x, save_dict=None, lambda_dict=None, p=1, quant_params=None):
        # EWGS Logic
        x = (x - self.lF) / (self.uF - self.lF)
        x = x.clamp(min=0, max=1) # [0, 1]

        if not self.baseline:
            if save_dict:
                save_dict["type"] = "feature"
            x = self.EWGS_discretizer(x, self.feature_levels, self.bkwd_scaling_factorF, save_dict, None, None)
        else:
            x = self.STE_discretizer(x, self.feature_levels)

        if self.hook_Fvalues:
            self.buff_feature = x
            self.buff_feature.retain_grad()

        if self.distill_pos:
            x = x * self.output_scale
        
        return x

    def initialize(self, x):
        self.uF.data.fill_(x.std() / math.sqrt(1 - 2/math.pi) * 3.0)
        self.lF.data.fill_(x.min())

        print(f"Initialized Feature Quantizer")
    
    def forward(self, x, save_dict=None, lambda_dict=None, quant_params=None):
        if self.init == 1:
            self.initialize(x)
        
        if self.use_student_quant_params and quant_params is not None:
            self.lF.data.fill_(quant_params.get("lA", self.lF.item()))
            self.uF.data.fill_(quant_params.get("uA", self.uF.item()))

        output = self.feature_quantization(x, save_dict, lambda_dict, quant_params=quant_params)

        return output