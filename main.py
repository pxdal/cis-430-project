# fix for "forrtl: error (200): program aborting due to control-C event"
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import sys
import torch
import numpy as np
from crowds_data import HumanTrajectory, VSPDatasetLoader, VSPDataset, VSPDataLoader
from noise_predictors import BasicNoisePredictor
from diffusion import DDPM
import matplotlib.pyplot as plt

def format_batch(batch, ddpm):
    labels, conditioning = batch
    
    formatted_labels = []
    
    for label in labels:
        inp = ddpm.noise_predictor.format_input(label)
        
        formatted_labels.append(inp)
    
    formatted_conditioning = []
    
    for (root_data, agent_data, agent_includes) in conditioning:
        condition = ddpm.noise_predictor.format_conditioning_single(root_data, agent_data, agent_includes)
        
        formatted_conditioning.append(condition)
    
    formatted_labels = torch.stack(formatted_labels)
    formatted_conditioning = torch.stack(formatted_conditioning)
    
    return formatted_labels, formatted_conditioning

def get_next_formatted_batch(dataloader, ddpm):
    return format_batch(next(iter(dataloader)), ddpm)

# train provided diffusion model on provided dataset
def train(model, dataloader, num_epochs, learning_rate, logging_rate=10):
    model.train()
    
    optimizer = torch.optim.Adam(model.noise_predictor.parameters(), lr=learning_rate)
    
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
    torch.set_printoptions(sci_mode=False)
    
    # get default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    dataset = VSPDatasetLoader(vsp_dir="data_zara", vsp_name="crowds_zara01.vsp", frame_size=(720, 576)).load()
    
    # model settings
    timesteps = 100
    in_steps = 50
    out_steps = 100
    num_agents = 5
    
    # create diffusion model
    ddpm = DDPM(timesteps, in_steps, out_steps, num_agents, device=device, noise_predictor="unet")
    
    # create dataloader (manages batching and shuffling)
    dataloader = VSPDataLoader(dataset, in_steps, out_steps, num_agents, batch_size=64)
    
    # train model
    # ddpm.load_checkpoint("adv_unet_checkpoint.pth")
    train(ddpm, dataloader, num_epochs=500, learning_rate=2e-4)
    ddpm.save_checkpoint("adv_unet_checkpoint.pth")
    
    # get a test sample
    # TODO: I gotta jump through a lot of hoops just to get a single sample...
    test_labels_all, test_conditioning_all = next(iter(dataloader))
    
    test_labels = [test_labels_all[0]]
    test_conditioning = [test_conditioning_all[0]]
    
    test_labels, test_conditioning = format_batch((test_labels, test_conditioning), ddpm)
    
    # run inference on test conditioning
    ddpm.eval()
    
    with torch.no_grad():
        predicted_trajectory = ddpm.backward_single(test_conditioning.squeeze(0))
        
        predicted_trajectory = predicted_trajectory.reshape(-1, 2).cpu()
        actual_trajectory = test_labels[0].reshape(-1, 2).cpu()

        predicted_trajectory = list(predicted_trajectory)
        actual_trajectory = list(actual_trajectory)
        
        predicted_trajectory = HumanTrajectory([], predicted_trajectory)
        actual_trajectory = HumanTrajectory([], actual_trajectory)
        
        predicted_trajectory = predicted_trajectory.format_as_positions()
        actual_trajectory = actual_trajectory.format_as_positions()
        
        predicted_trajectory = np.array(predicted_trajectory.positions)
        actual_trajectory = np.array(actual_trajectory.positions)
        
        print(predicted_trajectory)
        print(actual_trajectory)
        
        _, ax = plt.subplots()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        ax.plot(actual_trajectory[:, 0], actual_trajectory[:, 1])
        
        _, ax2 = plt.subplots()
        ax2.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1])
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        
        plt.show()
    
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)