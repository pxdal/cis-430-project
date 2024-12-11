import numpy as np
import torch

# noise predictor used for the trajectory denoiser.  this is a very barebones linear implementation.  it doesn't use time embeddings, or at all include timestep as an input
class BasicNoisePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # define model
        self.model = torch.nn.Sequential(
        )
    
    def forward(self, x):
        return self.model(x)