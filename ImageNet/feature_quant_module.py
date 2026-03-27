import torch
import torch.nn as nn
from custom_modules import EWGS_discretizer

class FeatureQuantizer(nn.Module):
    def __init__(self, args):
        super(FeatureQuantizer, self).__init__()
        self.feature_levels = args.feature_levels
        self.baseline = args.baseline
        self.use_student_quant_params = args.use_student_quant_params

        self.EWGS_discretizer = EWGS_discretizer.apply

        self.uF = nn.Parameter(data = torch.tensor(1).float(), requires_grad = not args.use_student_quant_params)
        self.lF = nn.Parameter(data = torch.tensor(0).float(), requires_grad = not args.use_student_quant_params)
        self.register_buffer('bkwd_scaling_factorF', torch.tensor(args.bkwd_scaling_factorF).float())

        self.register_buffer('init', torch.tensor(0))
    
    def feature_quantization(self, x):
        x = (x - self.lF) / (self.uF - self.lF)
        x = x.clamp(min=0, max=1) # [0, 1]

        if not self.baseline:
            x = self.EWGS_discretizer(x, self.feature_levels, self.bkwd_scaling_factorF)

        return x

    def forward(self, x, quant_params=None):
        if self.use_student_quant_params and quant_params is not None:
            self.lF.data.fill_(quant_params.get("lA", self.lF.item()))
            self.uF.data.fill_(quant_params.get("uA", self.uF.item()))

        output = self.feature_quantization(x)

        return output