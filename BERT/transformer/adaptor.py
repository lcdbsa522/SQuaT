import torch
import torch.nn as nn

class AdaptorBERT(nn.Module):
    def __init__(self, hidden_size, use_bn=False):
        super(AdaptorBERT, self).__init__()
        self.use_bn = use_bn
        
        self.linear = nn.Linear(hidden_size, hidden_size)
        
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_size)
        else:
            self.bn = None
            
    def forward(self, x):
        original_shape = x.shape
        
        if len(x.shape) == 3:
            B, seq_len, hidden_size = x.shape
            x = x.view(B * seq_len, hidden_size)
            need_reshape = True
        else:
            need_reshape = False
        
        x = self.linear(x)
        
        if self.bn is not None:
            x = self.bn(x)
        
        if need_reshape:
            x = x.view(original_shape)
        
        return x

