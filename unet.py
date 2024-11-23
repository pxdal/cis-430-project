import numpy as np
import torch

# intermediate convolutional layers - don't change size of data, just the number of feature layers
class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same"),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

# downsamples data to 1/2 and applies convolution    
class DownConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # first we downscale by a factor of 2,
        # then we apply a standard convolution to get output channels
        self.down_conv = torch.nn.Sequential(
            torch.nn.MaxPool1d(kernel_size=2),
            Conv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.down_conv(x)

# upscales data by 2 and applies convolution, also applies skip connections
class UpConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # we have the transpose operation output half the input because the skip connections add the other half of the channels
        # then the convolution operation outputs the proper number of channels
        self.up_conv = torch.nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = Conv(in_channels, out_channels)
    
    def forward(self, x, skip):
        # get upscaled layers
        upscaled = self.up_conv(x)
        
        # concatenate with skip connection
        x = torch.cat([skip, upscaled], dim=1)
        
        return self.conv(x)

# unet implementation
class UNet1d(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        # input layer converts input to feature layers
        self.inp = Conv(input_channels, 16)
        
        self.down1 = DownConv1d(16, 32)
        self.down2 = DownConv1d(32, 64)
        
        self.up1 = UpConv1d(64, 32)
        self.up2 = UpConv1d(32, 16)
        
        # output layer converts feature layers to output channels
        self.out = Conv(16, output_channels)
    
    def forward(self, x):
        x = self.inp(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        
        u1 = self.up1(d2, d1)
        u2 = self.up2(u1, x)
        
        return self.out(u2)
