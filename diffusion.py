# basic implementation of a diffusion model

import numpy as np
import torch

# class for training + inferencing with diffusion trajectory predictor
class TrajectoryPredictorDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def train(self):
        # going off of the training process detailed in the original ddpm paper
        
        
        return
    
    def forward(self, x):
        return x