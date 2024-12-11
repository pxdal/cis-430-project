import sys
import torch
import numpy as np
from crowds_data import HumanTrajectory, VSPDatasetLoader, VSPDataset
from noise_predictors import BasicNoisePredictor
from diffusion import DDPM

# train provided diffusion model on provided dataset
def train(model, dataset):
    return

def main(argc, argv):
    # get default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    data = VSPDatasetLoader(vsp_dir="data_zara", vsp_name="crowds_zara01.vsp", frame_size=(720, 576)).load()
    
    # model settings
    timesteps = 1000
    in_steps = 50
    out_steps = 100
    num_agents = 5
    
    # create diffusion model
    ddpm = DDPM(timesteps, in_steps, out_steps, num_agents, device=device, noise_predictor="basic")
    
    
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)