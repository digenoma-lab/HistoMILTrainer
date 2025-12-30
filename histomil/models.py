
"""Extracted from CLAM and TRIDENT"""
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from src.builder import create_model

def import_model(model_name, pretrained_model):
    return create_model(f"{model_name}.base.{pretrained_model}")
