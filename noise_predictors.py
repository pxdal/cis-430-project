# this file contains "noise predictor" implementations - the backbone of diffusion models used to predict the noise added to the input.

import numpy as np
import torch

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
            
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            
            torch.nn.Linear(256, self.output_size)
        ).to(self.device)
    
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