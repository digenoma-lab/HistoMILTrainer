
"""Extracted from CLAM and TRIDENT"""
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from trident.slide_encoder_models import ABMILSlideEncoder

class ABMIL(nn.Module):
    def __init__(self, input_feature_dim=1536, n_heads=1, head_dim=512, dropout=0., gated=True, hidden_dim=256, attention_reg_weight=0.01):
        super().__init__()
        self.attention_reg_weight = attention_reg_weight
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
        
        # Initialize attention weights for uniform distribution
        self._initialize_uniform_attention()
    
    def _initialize_uniform_attention(self):
        """
        Initialize branching layers to produce uniform attention scores.
        By setting weights to zero and bias to a constant, all patches will have
        the same logit value, resulting in uniform attention after softmax.
        """
        # Access the image_pooler (ABMIL module) inside feature_encoder
        image_pooler = self.feature_encoder.model['image_pooler']
        
        # Initialize each branching layer to produce constant outputs
        for branching_layer in image_pooler.branching_layers:
            # Set weights to very small values (near zero)
            nn.init.zeros_(branching_layer.weight)
            # Set bias to a small constant value (will be normalized by softmax)
            # Using a small positive value ensures uniform distribution after softmax
            nn.init.constant_(branching_layer.bias, 0.0)
    
    def compute_attention_regularization(self, attention_scores_softmax):
        """
        Compute L2 regularization penalty for attention scores.
        Penalizes deviation from uniform distribution (1/num_patches for each patch).
        
        Args:
            attention_scores_softmax: Attention scores after softmax.
                Shape: [batch_size, n_branches, n_heads, num_patches] or similar
        
        Returns:
            Regularization loss (scalar)
        """
        # Flatten to [batch_size, n_branches, n_heads, num_patches]
        if attention_scores_softmax.dim() == 4:
            batch_size, n_branches, n_heads, num_patches = attention_scores_softmax.shape
            # Expected uniform value: 1.0 / num_patches
            uniform_value = 1.0 / num_patches
            
            # Compute L2 penalty: sum of squared deviations from uniform
            # Shape: [batch_size, n_branches, n_heads]
            l2_penalty = torch.sum((attention_scores_softmax - uniform_value) ** 2, dim=-1)
            
            # Average over batch, branches, and heads
            reg_loss = torch.mean(l2_penalty)
        else:
            # Fallback: compute variance of attention scores
            # Higher variance = less uniform = higher penalty
            reg_loss = torch.var(attention_scores_softmax, dim=-1).mean()
        
        return reg_loss

    def forward(self, x, return_raw_attention=False, return_attention_reg=False):
        """
        Forward pass with optional attention regularization.
        
        Args:
            x: Input features (dict with 'features' key or tensor)
            return_raw_attention: If True, return attention scores
            return_attention_reg: If True, return attention regularization loss
        
        Returns:
            logits: Classification logits
            attention_scores (optional): Raw attention scores from ABMIL
            reg_loss (optional): Attention regularization loss
        """
        # Prepare input for ABMILSlideEncoder
        if isinstance(x, dict):
            batch = x
        else:
            batch = {'features': x}
        
        # Forward through feature encoder with attention scores if needed
        need_attention = return_raw_attention or return_attention_reg
        if need_attention:
            features, attention_scores = self.feature_encoder(batch, return_raw_attention=True)
            # attention_scores shape: [batch_size, n_branches, n_heads, num_patches]
        else:
            features = self.feature_encoder(batch)
            attention_scores = None
        
        # Classification
        logits = self.classifier(features)  # Shape: [batch_size, 2]
        
        # Compute regularization loss if requested
        attention_reg_loss = None
        if return_attention_reg and attention_scores is not None:
            attention_reg_loss = self.compute_attention_regularization(attention_scores)
        
        # Return based on flags
        if return_raw_attention and return_attention_reg:
            return logits, attention_scores, attention_reg_loss
        elif return_raw_attention:
            return logits, attention_scores
        elif return_attention_reg:
            return logits, attention_reg_loss
        else:
            return logits
    