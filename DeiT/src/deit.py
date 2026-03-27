import torch
import torch.nn as nn
from functools import partial

from .deit_vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224'
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
    
    def reset_classifier(self, num_classes, global_pool=''):
        super().reset_classifier(num_classes, global_pool)
        if self.head_dist is not None:
            self.head_dist = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, is_feat=False, quant_params=None):
        return super().forward_features(x, is_feat=is_feat, quant_params=quant_params)

    def forward(self, x, is_feat=False, quant_params=None):
        return super().forward(x, is_feat=is_feat, quant_params=quant_params)


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    target_num_classes = kwargs.get('num_classes', 1000)
    
    model_kwargs = {k: v for k, v in kwargs.items() 
                    if k not in ['pretrained_cfg', 'pretrained_cfg_overlay']}
    model_kwargs['num_classes'] = 1000
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer= nn.GELU, **model_kwargs)
    model.default_cfg = _cfg()
    
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        print("Loading ImageNet pretrained weights")
        model.load_state_dict(checkpoint["model"], strict=False)
    
    if target_num_classes != 1000:
        model.reset_classifier(target_num_classes)
    
    return model

@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    target_num_classes = kwargs.get('num_classes', 1000)
    
    model_kwargs = {k: v for k, v in kwargs.items() 
                    if k not in ['pretrained_cfg', 'pretrained_cfg_overlay']}
    model_kwargs['num_classes'] = 1000
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer= nn.GELU, **model_kwargs)
    model.default_cfg = _cfg()
    
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        print("Loading ImageNet pretrained weights")
        model.load_state_dict(checkpoint["model"], strict=False)
    
    if target_num_classes != 1000:
        model.reset_classifier(target_num_classes)
    
    return model

