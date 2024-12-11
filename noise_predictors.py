# this file contains various "noise predictor" implementations - the backbone of diffusion models used to predict the noise added to the input.

# for modularity, all noise predictors should include a function format_input(root_data, agent_data, agent_includes) such that the following call:
#   noise_predictor(noisy_input, noise_predictor.format_conditioning(root_data, agent_data, agent_includes))
# would be valid for any noise_predictor.  we do this rather than enforcing parameters so that the conditioning can be formatted once and used multiple times.

# all of these should accept as input to forward the output of VSTDataset().get_training_sample for modularity.  Specifically, the method signature should be:
    # forward(noisy_input, root_data, agent_data, agent_includes)

import numpy as np
import torch

# noise predictor used for the trajectory denoiser.  this is a very barebones linear implementation.
class BasicNoisePredictor(torch.nn.Module):
    def __init__(self, in_steps, out_steps, num_agents):
        super().__init__()
        
        # calculate output size
        # two nodes (x, y) per output step for the root trajectory
        self.output_size = 2*out_steps
        
        # calculate input size
        # output size for the noisy input
        # two nodes (x, y) per input step for the input trajectory
        # three nodes (x, y, include) per step per input agent 
        self.input_size = self.output_size + 2*in_steps + 3*in_steps*num_agents
        
        # define model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 512),
            torch.nn.ReLU(),
            
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            
            torch.nn.Linear(256, self.output_size)
        )
    
    # TODO: it's particularly inefficient to do this for inference because the input has to be recreated for every timestep.
    def forward(self, noisy_input, root_data, agent_data, agent_includes):
        # pack root to single np array
        root_positions = np.array(root_data.positions)
        root_packed = root_positions.flatten()
        
        agents_packed = []
        
        # pack agents to np arrays
        for agent_trajectory, agent_include in zip(agent_data, agent_includes):
            positions = np.array(agent_trajectory.positions).flatten()
            includes = np.array(agent_include)
            
            agents_packed.append(np.concatenate((positions, includes)))
        
        # concatenate all agents
        agents_packed = np.concatenate(agents_packed)
        
        x = torch.tensor(np.concatenate((root_packed, agents_packed)))
        
        x = torch.cat((noisy_input, x))
        x = x.float()
        
        return self.model(x)