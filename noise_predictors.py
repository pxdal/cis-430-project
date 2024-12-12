# this file contains "noise predictor" implementations - the backbone of diffusion models used to predict the noise added to the input.

import math
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

class TimeEmbedding(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        
        self.n_channels = n_channels
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.n_channels // 4, self.n_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(self.n_channels, self.n_channels)
        )

    def forward(self, t):
        # create sinusoidal position embeddings
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.mlp(emb)
        
        return emb
        
class ResidualBlock(torch.nn.Module):
    # input_size, conditioning_size
    def __init__(self, in_channels, out_channels, output_size, conditioning_channels, conditioning_size, time_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.conditioning_channels = conditioning_channels
        self.conditioning_size = conditioning_size
        
        # converts conditioning to the proper shape
        # self.compress_conditioning = torch.nn.Conv1d(conditioning_channels, 1, kernel_size=3, padding="same")
        self.remap_conditioning = torch.nn.Linear(conditioning_size, output_size)
        
        self.conv1 = torch.nn.Sequential(
            # torch.nn.Conv1d(in_channels+1, out_channels, kernel_size=3, padding="same"),
            torch.nn.Conv1d(in_channels+conditioning_channels, out_channels, kernel_size=3, padding="same"),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.SiLU()
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same"),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.SiLU()
        )
        
        # map time embedding to output shape
        self.time_embedding_map = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(time_channels, out_channels)
        )
        
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same")
        else:
            self.shortcut = torch.nn.Identity()
    
    # t = timestep embedding
    def forward(self, x, t, conditioning):
        # # map conditioning from (batch_size, conditioning_channels, conditioning_size) to (batch_size, 1, conditioning_size)
        # conditioning = self.compress_conditioning(conditioning)
        # # map conditioning from (batch_size, 1, conditioning_size) to (batch_size, 1, output_size)
        # conditioning = self.remap_conditioning(conditioning)
        
        conditioning = self.remap_conditioning(conditioning)
        
        # shape is (batch_size, in_channels+1, output_size)
        h = torch.cat((x, conditioning), dim=1)
        
        h = self.conv1(h)
        
        # time embedding
        # shape here is (batch_size, out_channels, output_size)
        # time embedding should be shaped (batch_size, out_channels, 1)
        h += self.time_embedding_map(t).unsqueeze(-1)
        
        h = self.conv2(h)
        
        return h + self.shortcut(x)
    
# downsamples data to 1/2 and applies convolution    
class DownConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, output_size, conditioning_channels, conditioning_size, time_channels):
        super().__init__()
        
        # first we downscale by a factor of 2,
        # then we apply a standard convolution to get output channels
        self.down = torch.nn.MaxPool1d(kernel_size=2)
        self.res_block = ResidualBlock(in_channels, out_channels, output_size, conditioning_channels, conditioning_size, time_channels)
        
    def forward(self, x, t, conditioning):
        x = self.down(x)
        x = self.res_block(x, t, conditioning)
        
        return x

# upscales data by 2 and applies convolution, also applies skip connections
class UpConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, output_size, conditioning_channels, conditioning_size, time_channels):
        super().__init__()
        
        # we have the transpose operation output half the input because the skip connections add the other half of the channels
        # then the convolution operation outputs the proper number of channels
        
        self.up_conv = torch.nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(in_channels, out_channels, output_size, conditioning_channels, conditioning_size, time_channels)
    
    def forward(self, x, skip, t, conditioning):
        # get upscaled layers
        x = self.up_conv(x)
        
        # concatenate with skip connection
        x = torch.cat([skip, x], dim=1)
        
        return self.res_block(x, t, conditioning)

class UNetNoisePredictor(torch.nn.Module):
    def __init__(self, in_steps, out_steps, num_agents, device):
        super().__init__()
        
        self.input_channels = 1
        self.output_channels = 1
        
        # 1 for root trajectory, 2 for each agent
        self.conditioning_channels = 1 + num_agents*2
        # size of input trajectory
        self.conditioning_size = 2*in_steps
        
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.num_agents = num_agents
        self.device = device
        
        # required for diffusion sampling
        self.output_size = 2*self.out_steps
        self.input_size = self.output_size
        
        # should be whatever number of channels in the feature layer * 4
        self.time_channels = 32 * 4
        
        # input layer converts input to feature layers
        current_output_size = self.output_size
        
        # self.inp = ResidualBlock(self.input_channels, 32, current_output_size, self.conditioning_channels, self.conditioning_size)
        self.inp = torch.nn.Conv1d(self.input_channels, 32, kernel_size=3, padding="same")
        
        # downsamples data
        current_output_size //= 2
        self.down1 = DownConv1d(32, 64, current_output_size, self.conditioning_channels, self.conditioning_size, self.time_channels)
        
        current_output_size //= 2
        self.down2 = DownConv1d(64, 128, current_output_size, self.conditioning_channels, self.conditioning_size, self.time_channels)

        # do some processing in the middle
        self.middle1 = ResidualBlock(128, 128, current_output_size, self.conditioning_channels, self.conditioning_size, self.time_channels)
        
        # upsamples data + apply skip connections
        current_output_size *= 2
        self.up1 = UpConv1d(128, 64, current_output_size, self.conditioning_channels, self.conditioning_size, self.time_channels)
        
        current_output_size *= 2
        self.up2 = UpConv1d(64, 32, current_output_size, self.conditioning_channels, self.conditioning_size, self.time_channels)
        
        # output layer converts feature layers to output channels
        self.out = torch.nn.Sequential(
            torch.nn.Conv1d(32, self.output_channels, kernel_size=3, padding="same")
        )
        
        # time embedding layer
        self.time_embedding = TimeEmbedding(self.time_channels)
    
    def get_input_shape(self):
        return (1, self.input_size)
    
    def get_output_shape(self):
        return (1, self.output_size)
    
    def format_input(self, root_label):
        root_positions = np.array(root_label.positions).flatten()
        
        out = torch.tensor(root_positions).float().unsqueeze(0).to(self.device)
        
        return out
    
    def format_conditioning_single(self, root_trajectory, agent_trajectories, agent_includes):
        # conditioning for the unet is left unpacked so that each condition (root trajectory, agent trajectories, agent includes) get their own channel.
        
        # pack root to single np array
        root_positions = np.array(root_trajectory.positions)
        root_packed = root_positions.flatten()
        
        agent_tensor_trajectories = []
        agent_tensor_includes = []
        
        # pack agents to np arrays
        for agent_trajectory, agent_include in zip(agent_trajectories, agent_includes):
            positions = np.array(agent_trajectory.positions).flatten()
            includes = np.array(agent_include)
            
            agent_tensor_trajectories.append(torch.tensor(positions))
            agent_tensor_includes.append(torch.tensor(includes))
        
        # (num_agents, trajectory_length)
        agent_tensor_trajectories = torch.stack(agent_tensor_trajectories)
        # (num_agents, trajectory_length)
        agent_tensor_includes = torch.stack(agent_tensor_includes)
        
        # (1, trajectory_length)
        root_tensor = torch.tensor(root_packed).unsqueeze(0)
        
        c = torch.cat((root_tensor, agent_tensor_trajectories, agent_tensor_includes)).float().to(self.device)
        
        return c
    
    def forward(self, noisy_input, timestep, conditioning):
        t = self.time_embedding(timestep)
        
        # x = self.inp(noisy_input, conditioning)
        x = self.inp(noisy_input)
        
        d1 = self.down1(x, t, conditioning)
        d2 = self.down2(d1, t, conditioning)
        
        m1 = self.middle1(d2, t, conditioning)
        
        u1 = self.up1(m1, d1, t, conditioning)
        u2 = self.up2(u1, x, t, conditioning)
        
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
        # four nodes (x, y, include_x, include_y) per step per input agent 
        self.input_size = self.output_size + 2*self.in_steps + 4*self.in_steps*self.num_agents
        
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