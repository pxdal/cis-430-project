# this file contains "noise predictor" implementations - the backbone of diffusion models used to predict the noise added to the input.

import numpy as np
import torch

# unet building blocks...

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same"),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.SiLU()
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same"),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.SiLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same"),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.SiLU()
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same"),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.SiLU()
        )
        
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same")
        else:
            self.shortcut = torch.nn.Identity()
    
    def forward(self, x, conditioning):
        h = self.conv1(x)
        
        # add embedding...
        
        h = self.conv2(x)
        
        return h + self.shortcut(x)
    
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
        x = self.up_conv(x)
        
        # concatenate with skip connection
        x = torch.cat([skip, x], dim=1)
        
        return self.conv(x)

class UNetNoisePredictor(torch.nn.Module):
    def __init__(self, in_steps, out_steps, num_agents, device):
        super().__init__()
        
        self.input_channels = 1
        self.output_channels = 1
        
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.num_agents = num_agents
        self.device = device
        
        # required for diffusion sampling
        self.output_size = 2*self.out_steps
        self.input_size = self.output_size
        
        # input layer converts input to feature layers
        self.inp = ResidualBlock(self.input_channels, 32)
        
        # downsamples data
        self.down1 = DownConv1d(32, 64)
        self.down2 = DownConv1d(64, 128)

        # upsamples data + apply skip connections
        self.up1 = UpConv1d(128, 64)
        self.up2 = UpConv1d(64, 32)
        
        # output layer converts feature layers to output channels
        self.out = torch.nn.Sequential(
            torch.nn.Conv1d(32, self.output_channels, kernel_size=3, padding="same")
        )
    
    def get_input_shape(self):
        return (1, self.input_size)
    
    def get_output_shape(self):
        return (1, self.output_size)
    
    def format_input(self, root_label):
        root_positions = np.array(root_label.positions).flatten()
        
        out = torch.tensor(root_positions).float().unsqueeze(0).to(self.device)
        
        return out
    
    def format_conditioning_single(self, root_trajectory, agent_trajectories, agent_includes):
        # pack root to single np array
        root_positions = np.array(root_trajectory.positions)
        root_packed = root_positions.flatten()
        
        agents_packed = []
        
        # pack agents to np arrays
        for agent_trajectory, agent_include in zip(agent_trajectories, agent_includes):
            positions = np.array(agent_trajectory.positions).flatten()
            includes = np.array(agent_include)
            
            agents_packed.append(np.concatenate((positions, includes)))
        
        # concatenate all agents
        agents_packed = np.concatenate(agents_packed)
        
        c = torch.tensor(np.concatenate((root_packed, agents_packed)))
        
        return c.float().to(self.device)
    
    def forward(self, noisy_input, timestep, conditioning):
        x = self.inp(noisy_input)
        
        d1 = self.down1(x)
        d2 = self.down2(d1)

        u1 = self.up1(d2, d1)
        u2 = self.up2(u1, x)
        
        return self.out(u2)

# a simplified implementation of a unet, no resnet blocks or fancy timestep embeddings.
class BasicUNetNoisePredictor(torch.nn.Module):
    def __init__(self, in_steps, out_steps, num_agents, device):
        super().__init__()
        
        self.input_channels = 1
        self.output_channels = 1
        
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.num_agents = num_agents
        self.device = device
        
        # required for diffusion sampling
        self.output_size = 2*self.out_steps
        self.input_size = self.output_size
        
        # input layer converts input to feature layers
        self.inp = Conv(self.input_channels, 32)
        
        # downsamples data
        self.down1 = DownConv1d(32, 64)
        self.down2 = DownConv1d(64, 128)

        # upsamples data + apply skip connections
        self.up1 = UpConv1d(128, 64)
        self.up2 = UpConv1d(64, 32)
        
        # output layer converts feature layers to output channels
        self.out = torch.nn.Sequential(
            torch.nn.Conv1d(32, self.output_channels, kernel_size=3, padding="same")
        )
    
    def get_input_shape(self):
        return (1, self.input_size)
    
    def get_output_shape(self):
        return (1, self.output_size)
    
    def format_input(self, root_label):
        root_positions = np.array(root_label.positions).flatten()
        
        out = torch.tensor(root_positions).float().unsqueeze(0).to(self.device)
        
        return out
    
    def format_conditioning_single(self, root_trajectory, agent_trajectories, agent_includes):
        # pack root to single np array
        root_positions = np.array(root_trajectory.positions)
        root_packed = root_positions.flatten()
        
        agents_packed = []
        
        # pack agents to np arrays
        for agent_trajectory, agent_include in zip(agent_trajectories, agent_includes):
            positions = np.array(agent_trajectory.positions).flatten()
            includes = np.array(agent_include)
            
            agents_packed.append(np.concatenate((positions, includes)))
        
        # concatenate all agents
        agents_packed = np.concatenate(agents_packed)
        
        c = torch.tensor(np.concatenate((root_packed, agents_packed)))
        
        return c.float().to(self.device)
    
    def forward(self, noisy_input, timestep, conditioning):
        x = self.inp(noisy_input)
        
        d1 = self.down1(x)
        d2 = self.down2(d1)

        u1 = self.up1(d2, d1)
        u2 = self.up2(u1, x)
        
        return self.out(u2)
    

# this is a very barebones linear implementation.
class BasicNoisePredictor(torch.nn.Module):
    def __init__(self, in_steps, out_steps, num_agents, device):
        super().__init__()
        
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.num_agents = num_agents
        self.device = device
        
        # calculate output size
        # two nodes (x, y) per output step for the root trajectory
        self.output_size = 2*self.out_steps
        
        # calculate input size
        # output size for the noisy input
        # two nodes (x, y) per input step for the input trajectory
        # three nodes (x, y, include) per step per input agent 
        self.input_size = self.output_size + 2*self.in_steps + 3*self.in_steps*self.num_agents
        
        # define model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            
            torch.nn.Linear(256, self.output_size)
        ).to(self.device)
    
    def get_input_shape(self):
        return (self.input_size,)
    
    def get_output_shape(self):
        return (self.output_size,)
        
    def format_input(self, root_label):
        root_positions = np.array(root_label.positions)
        
        return torch.tensor(root_positions.flatten()).float().to(self.device)
        
    def format_conditioning_single(self, root_trajectory, agent_trajectories, agent_includes):
        # pack root to single np array
        root_positions = np.array(root_trajectory.positions)
        root_packed = root_positions.flatten()
        
        agents_packed = []
        
        # pack agents to np arrays
        for agent_trajectory, agent_include in zip(agent_trajectories, agent_includes):
            positions = np.array(agent_trajectory.positions).flatten()
            includes = np.array(agent_include)
            
            agents_packed.append(np.concatenate((positions, includes)))
        
        # concatenate all agents
        agents_packed = np.concatenate(agents_packed)
        
        c = torch.tensor(np.concatenate((root_packed, agents_packed)))
        
        return c.float().to(self.device)
    
    def forward(self, noisy_input, timesteps, conditioning):
        x = torch.cat((noisy_input, conditioning), dim=1)
        x = x.float()
        
        x = x.to(self.device)
        
        return self.model(x)