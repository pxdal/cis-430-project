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
    # load dataset
    data = VSPDatasetLoader(vsp_dir="data_zara", vsp_name="crowds_zara01.vsp", frame_size=(720, 576)).load()
    
    # model settings
    in_steps = 50
    out_steps = 100
    num_agents = 5

    predictor = BasicNoisePredictor(in_steps, out_steps, num_agents)
    
    (root_data, root_include, agent_data, agent_includes), (root_label, agent_labels) = data.get_training_sample(0, in_steps, out_steps, num_agents=num_agents, offset=0)
    
    out = predictor(torch.randn(2*out_steps), root_data, agent_data, agent_includes)
    
    # create diffusion model
    ddpm = DDPM(in_steps, out_steps, num_agents, noise_predictor="basic")
    
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)