import sys
import torch
import numpy as np
from crowds_data import HumanTrajectory, VSPDatasetLoader, VSPDataset, VSPDataLoader
from noise_predictors import BasicNoisePredictor
from diffusion import DDPM

def get_next_formatted_batch(dataloader, ddpm):
    labels, conditioning = next(iter(dataloader))
    
    formatted_labels = []
    
    for label in labels:
        inp = ddpm.noise_predictor.format_input(label)
        
        formatted_labels.append(inp)
    
    formatted_conditioning = []
    
    for (root_data, agent_data, agent_includes) in conditioning:
        condition = ddpm.noise_predictor.format_conditioning_single(root_data, agent_data, agent_includes)
        
        if condition.shape[0] != 850:
            print(root_data.frames)
            print(len(agent_data))
            print(agent_includes)
            print(condition)
            print(condition.shape)
        
        formatted_conditioning.append(condition)
    
    formatted_labels = torch.stack(formatted_labels)
    formatted_conditioning = torch.stack(formatted_conditioning)
    
    return formatted_labels, formatted_conditioning

# train provided diffusion model on provided dataset
def train(model, dataloader, num_epochs, learning_rate, logging_rate=10):
    ddpm.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for i in range(num_epochs):
        batch_trajectories, batch_conditioning = get_next_formatted_batch(dataloader, model)
        
        # get model loss for this epoch
        loss = model.loss(batch_trajectories, batch_conditioning)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % logging_rate == 0:        
            print("Epoch:", i, " - Loss:", float(loss))
        
    return

def main(argc, argv):
    # get default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    dataset = VSPDatasetLoader(vsp_dir="data_zara", vsp_name="crowds_zara01.vsp", frame_size=(720, 576)).load()
    
    # model settings
    timesteps = 1000
    in_steps = 50
    out_steps = 100
    num_agents = 5
    
    # create diffusion model
    ddpm = DDPM(timesteps, in_steps, out_steps, num_agents, device=device, noise_predictor="basic")
    
    # create dataloader (manages batching and shuffling)
    dataloader = VSPDataLoader(dataset, in_steps, out_steps, num_agents, batch_size=64)
    
    # train model
    train(ddpm, dataloader, num_epochs=100, learning_rate=1e-3)
    
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)