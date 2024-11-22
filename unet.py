import numpy as np
import torch

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same"),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)
    
class DownConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.down_conv = torch.nn.Sequential(
            Conv(in_channels, out_channels),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        
    def forward(self, x):
        return self.down_conv(x)

class UpConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels, out_channels),
            
        )
    
class UNet1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.down1 = DownConv1d(1, 2)
        self.down2 = DownConv1d(2, 4)
        self.down3 = DownConv1d(4, 8)
        
    
    def forward(self, x):
        print(x.dtype)
        out = self.down1(x)
        out = self.down2(out)
        out = self.down3(out)
        
        return out
