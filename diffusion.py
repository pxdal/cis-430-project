# basic implementation of a diffusion model
# goal is to have the noise predictor be modular, though this is difficult because the input format could be different

import numpy as np
import torch
from noise_predictors import BasicNoisePredictor

class DDPM():
    def __init__(self, in_steps, out_steps, num_agents, noise_predictor="basic"):
        if noise_predictor == "basic":
            self.noise_predictor = BasicNoisePredictor(in_steps, out_steps, num_agents)
        else:
            raise Exception("Nonexistent noise predictor \"" + noise_predictor + "\"")