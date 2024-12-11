import numpy as np
import torch

# implementation of the denoising unet we're using in the diffusion model itself
# based heavily off of this implementation from labml.ai: https://nn.labml.ai/diffusion/ddpm/unet.html



# unlike the original unet, the ddpm paper uses residual blocks as the building blocks of each layer (because they're neat!)
class ResidualBlock(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        num_groups=32
    ):
        super().__init__()
        
        self.block = torch.nn.Sequential(
            self.make_layer(in_channels, out_channels, num_groups),
            self.make_layer(out_channels, out_channels, num_groups)
        )
        
        # if in_channels != out_channels, we use a conv layer to map the input to the proper shape
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same")
        else:
            self.shortcut = torch.nn.Identity()
        
    def make_layer(self, in_channels, out_channels, num_groups):
        return torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups, in_channels),
            torch.nn.SiLU(), # swish function is used instead of ReLU in many implementations I've seen for some reason
            torch.nn.Conv1d(input_channels, output_channels, kernel_size=3, padding="same")
        )
    
    def forward(self, x):
        # residuals
        g = self.block(x)
        
        # shortcut connection
        return self.shortcut(x) + g

# a key difference between ordinary unets and the noise predicting unet is the usage of time embeddings
# we embed the current timestep at each layer of the unet via concatenation
class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        
        self.n_channels = n_channels
        
        # some learned parameters applied to the embedding
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.n_channels // 4, self.n_channels),
            torch.nn.SiLU(), # swish
            torch.nn.Linear(self.n_channels, self.n_channels)
        )
    
    def forward(self, t):
        half_dim = self.n_channels // 8 # half of input channels to self.mlp
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb) # move to device
        