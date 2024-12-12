# basic implementation of a diffusion model

import numpy as np
import torch
from noise_predictors import BasicNoisePredictor, UNetNoisePredictor

# define noise schedulers
def linear_noise_schedule(timesteps):
    b_min = 0.0001
    b_max = 0.02
    
    return torch.linspace(b_min, b_max, timesteps)

class DDPM():
    noise_predictors = {
        "basic": BasicNoisePredictor,
        "unet": UNetNoisePredictor
    }
    
    noise_schedulers = {
        "linear": linear_noise_schedule
    }
    
    def __init__(self, timesteps, in_steps, out_steps, num_agents, device, noise_predictor="basic", noise_schedule="linear"):
        self.timesteps = timesteps
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.device = device
        
        # create noise predictor model
        if noise_predictor in self.noise_predictors:
            self.noise_predictor = self.noise_predictors[noise_predictor](in_steps, out_steps, num_agents, device).to(self.device)
        else:
            raise Exception("Nonexistent noise predictor \"" + noise_predictor + "\"")
        
        # get beta values (noise scheduling)
        if noise_schedule in self.noise_schedulers:
            self.beta = self.noise_schedulers[noise_schedule](self.timesteps).to(device)
        else:
            raise Exception("Nonexistent noise schedule \"" + noise_schedule + "\"")
        
        # get alpha and alpha bar
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, axis=0)
        
        # get variance at each timestep
        # (which is exactly what the noise schedule is, but for clarity it gets its own variable)
        self.sigma2 = self.beta
        
        self.loss_func = torch.nn.MSELoss()
    
    # get parameters for training
    def parameters(self):
        return self.noise_predictor.parameters()
    
    # get the mean and variance of the q(x_t | x_0) distribution for the batch of timesteps given the batch of x0
    # there should be as many timesteps as samples in the batch
    # mean shape: same as x0
    # var shape: same as t
    def get_q_distribution(self, x0, t):
        # mean is sqrt(alpha_bar) * x0
        alpha_bars = self.alpha_bar.gather(0, t)
        
        while len(alpha_bars.shape) < len(x0.shape):
            alpha_bars = alpha_bars.unsqueeze(-1)
        
        mean = alpha_bars ** 0.5 * x0
        
        var = 1 - self.alpha_bar.gather(0, t)
        
        return mean, var
    
    # sample from the q(x_t | x_0) distribution for the batch of timesteps each given the batch of x_0
    # this is the "forward process"
    def sample_q_distribution(self, x0, t, eps=None):
        mean, var = self.get_q_distribution(x0, t)
        
        if eps is None:
            eps = torch.randn_like(x0).to(self.device)
        
        # reparameterization trick
        std = var ** 0.5
        
        while len(std.shape) < len(x0.shape):
            std = std.unsqueeze(-1)
            
        return mean + std*eps
    
    # sample from the p(x_t-1 | x_t, c) distribution given a single xt, t, and conditioning
    def sample_p_distribution(self, xt, t, conditioning, eps=None):
        # get the noise prediction
        noise_prediction = self.noise_predictor(xt, t, conditioning)
        
        beta = self.beta[t]
        alpha = self.alpha[t]
        alpha_bar = self.alpha_bar[t]
        
        # calculate mean of distribution
        noise_coefficient = beta / ((1 - alpha_bar) ** 0.5)
        diff_coefficient = 1 / (alpha ** 0.5)
        
        mean = diff_coefficient * ( xt - (noise_coefficient * noise_prediction) )
        
        # get variance
        var = self.sigma2[t]
        
        # get added noise from variance
        if eps is None:
            eps = torch.randn_like(xt).to(self.device)
        
        # reparameterization trick
        std = var ** 0.5
        
        return mean + std*eps
    
    # predict a future trajectory from conditioning by sampling from the predicted distribution
    def backward_single(self, conditioning):    
        # start from pure noise
        shape = self.noise_predictor.get_output_shape()
        
        xt = torch.randn(shape).unsqueeze(0).to(self.device)
        conditioning = conditioning.unsqueeze(0)
        
        # iteratively denoise
        for t in reversed(range(self.timesteps)):
            if t == 0:
                z = torch.zeros(shape)
            else:
                z = torch.randn(shape)
            
            z = z.to(self.device)
            
            xt = self.sample_p_distribution(xt, torch.tensor([t]), conditioning)
        
        return xt[0]
    
    # get the model's loss for a batch of training samples with conditioning
    def loss(self, x0, conditioning, eps=None):
        # get random noise to be added
        if eps is None:
            eps = torch.randn_like(x0)
        
        # get random timesteps for each item in batch
        batch_size = x0.shape[0]
        
        t = torch.randint(0, self.timesteps, (batch_size,)).to(self.device)
        
        # forward process (with pre-chosen eps from above)
        xt = self.sample_q_distribution(x0, t, eps=eps)
        
        # get noise prediction
        eps_prediction = self.noise_predictor(xt, t, conditioning)
        
        # print(eps_prediction)
        # print(eps)
        # print()
        
        return self.loss_func(eps_prediction, eps)
    
    def train(self):
        self.noise_predictor.train()
    
    def eval(self):
        self.noise_predictor.eval()
    
    def save_checkpoint(self, checkpoint_dir):
        torch.save(self.noise_predictor.state_dict(), checkpoint_dir)
    
    def load_checkpoint(self, checkpoint_dir):
        self.noise_predictor.load_state_dict(torch.load(checkpoint_dir, weights_only=False))