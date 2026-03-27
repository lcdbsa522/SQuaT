from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillCosine(nn.Module):
    def __init__(self, eps=1e-8):
        super(FeatureDistillCosine, self).__init__()
        self.eps = eps

    def forward(self, feat_s, feat_t):
        """
        Args:
            feat_s: [N, C, H, W] student features
            feat_t: [N, C, H, W] teacher features
        """
        # Flatten to [N, CHW]
        N = feat_s.size(0)
        feat_s_flat = feat_s.view(N, -1)
        feat_t_flat = feat_t.view(N, -1)

        # Normalize (cosine similarity uses L2-normalized vectors)
        feat_s_norm = F.normalize(feat_s_flat, dim=1, eps=self.eps)
        feat_t_norm = F.normalize(feat_t_flat, dim=1, eps=self.eps)

        # Compute cosine similarity: [N]
        cos_sim = (feat_s_norm * feat_t_norm).sum(dim=1)

        # Loss = 1 - cosine similarity
        loss = (1 - cos_sim).mean()

        return loss