
"""Extracted from CLAM and TRIDENT"""
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from trident.slide_encoder_models import ABMILSlideEncoder

class ABMIL(nn.Module):
    def __init__(self, input_feature_dim=1536, n_heads=1, head_dim=512, dropout=0., gated=True, hidden_dim=256):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            input_feature_dim=input_feature_dim, 
            n_heads=n_heads, 
            head_dim=head_dim, 
            dropout=dropout, 
            gated=gated
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 outputs for binary classification
        )

    def forward(self, x, return_raw_attention=False):
        # ABMILSlideEncoder expects a dictionary with 'features' key
        if isinstance(x, dict):
            batch = x
        else:
            batch = {'features': x}
        
        if return_raw_attention:
            features, attn = self.feature_encoder(batch, return_raw_attention=True)
        else:
            features = self.feature_encoder(batch)
        logits = self.classifier(features)  # Shape: [batch_size, 2]
        if return_raw_attention:
            return logits, attn
        return logits
    