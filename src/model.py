import torch
import torch.nn as nn
import rtdl_revisiting_models as rtdl
import numpy as np
from typing import List, Optional


class MultiLabelFTTransformer(nn.Module):
    def __init__(self, n_num_features: int, cat_cardinalities: List[int], n_classes: int = 32):
        super().__init__()
        
        self.ftt = rtdl.FTTransformer(
            n_cont_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_out=128,
            n_blocks=3, 
            d_block=192, 
            attention_n_heads=8, 
            attention_dropout=0.2,
            ffn_d_hidden_multiplier=384,
            ffn_dropout=0.2,
            residual_dropout=0.0
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
        
        self.init_final_bias()

    def init_final_bias(self, prob: float = 0.05):
        prior_prob = prob
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.classifier[-1].bias, bias_value)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = self.ftt(x_num, x_cat)
        logits = self.classifier(x)
        return logits

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = -1 * (los_pos * torch.pow(1 - xs_pos, self.gamma_pos) + \
                     los_neg * torch.pow(1 - xs_neg, self.gamma_neg))
        return loss.sum()