from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillKL(nn.Module):
    def __init__(self, T):
        super(FeatureDistillKL, self).__init__()
        self.T = T
    
    def forward(self, feat_s, feat_t):
        """
        Args:
            feat_s: [N, C, H, W] student features
            feat_t: [N, C, H, W] teacher features
        """

        prob_s = self._feature_to_prob(feat_s)
        prob_t = self._feature_to_prob(feat_t)

        p_s = F.log_softmax(prob_s / self.T, dim=1)
        p_t = F.softmax(prob_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / feat_s.shape[0]

        return loss

    def _feature_to_prob(self, features):
        batch_size = features.size(0)
        features_flat = features.view(batch_size, -1)
        unique_values = torch.unique(features).sort()[0]
        num_bins = len(unique_values)

        if num_bins == 1:
            return torch.zeros(batch_size, 2, device=features.device)

        histogram = torch.zeros(batch_size, num_bins, device=features.device)

        for i, bin_value in enumerate(unique_values):
            counts = (features_flat == bin_value).float().sum(dim=1)
            histogram[:, i] = counts

        eps = 1e-8
        histogram = histogram + eps

        prob_logits = torch.log(histogram)

        return prob_logits