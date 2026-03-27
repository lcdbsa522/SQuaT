import torch.nn.functional as F

from src.deit_vision_transformer import Attention as deit_attention
from .qbias import LearnableBias
from .qlinear import QLinear
from ..quantizer.lsq import LsqQuantizer, LsqQuantizer4v


class QAttention(deit_attention):
    def __init__(self, m, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable=True,
                 weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized=False,
                 **kwargs):
        assert isinstance(m, deit_attention)
        
        qqkkvv = getattr(m, 'qqkkvv', False)
        
        super().__init__(
            dim=m.qkv.in_features,
            num_heads=m.num_heads,
            attn_drop=m.attn_drop.p,
            proj_drop=m.proj_drop.p,
            qqkkvv=qqkkvv
        )
        self.weight_bits = weight_bits
        self.input_bits = input_bits

        self.qkv = QLinear(
            m=self.qkv,
            weight_bits=weight_bits,
            input_bits=input_bits,
            weight_channelwise=weight_channelwise,
            input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
            aq_learnable=aq_learnable,
            wq_learnable=wq_learnable,
            symmetric=True,
            pretrained_initialized=pretrained_initialized
        )
        
        self.proj = QLinear(
            m=self.proj,
            weight_bits=weight_bits,
            input_bits=input_bits,
            weight_channelwise=weight_channelwise,
            input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
            aq_learnable=aq_learnable,
            wq_learnable=wq_learnable,
            symmetric=True,
            pretrained_initialized=pretrained_initialized
        )

        self.quan_a_q_fn = LsqQuantizer(bit=input_bits, all_positive=False, per_channel=True, learnable=aq_learnable)
        self.quan_a_k_fn = LsqQuantizer(bit=input_bits, all_positive=False, per_channel=True, learnable=aq_learnable)
        self.quan_a_v_fn = LsqQuantizer4v(bit=input_bits, all_positive=False, per_channel=True, learnable=aq_learnable)

        self.move_qkv_b4 = LearnableBias(m.qkv.in_features * 3)
        self.move_q_aft = LearnableBias(m.qkv.in_features)
        self.move_k_aft = LearnableBias(m.qkv.in_features)
        self.move_v_aft = LearnableBias(m.qkv.in_features)

        self.quan_a_softmax_fn = LsqQuantizer(bit=input_bits, all_positive=True, per_channel=True, learnable=aq_learnable)

    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x)
        
        if self.input_bits < 32:
            qkv = self.move_qkv_b4(qkv)
        
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = self.quan_a_q_fn(q)
        k = self.quan_a_k_fn(k)
        
        v = v.permute(0, 2, 1, 3).reshape(B, N, C)
        v = self.quan_a_v_fn(v)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.input_bits < 32:
            q = q.permute(0, 2, 1, 3).reshape(B, N, C)
            k = k.permute(0, 2, 1, 3).reshape(B, N, C)
            v = v.permute(0, 2, 1, 3).reshape(B, N, C)
            q = self.move_q_aft(q)
            k = self.move_k_aft(k)
            v = self.move_v_aft(v)
        
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn_weights = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn_prob = F.softmax(attn_weights, dim=-1)

        attn_prob = self.quan_a_softmax_fn(attn_prob)
        attn_prob = self.attn_drop(attn_prob)

        x = (attn_prob @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, None

