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

def convert_tensor_trajectory_to_plottable(initial, x, include_initial=True, scale=(1, 1)):
    if initial == None:
        x = x.reshape(-1, 2).cpu()
        x = x.tolist()
    else:
        initial = initial.reshape(-1, 2).cpu()
        initial = initial.tolist()
        
        x = x.reshape(-1, 2).cpu()
        x = x.tolist()
        
        if include_initial:
            x = initial + x
        else:
            initial = HumanTrajectory([], initial)
            initial = initial.format_as_positions()
            last = initial.positions[-1]
            x = [last] + x
    
    x = HumanTrajectory([], x)
    x = x.format_as_positions()
    x = np.array(x.positions)
    
    xs = x[:, 0] * scale[0]
    ys = x[:, 1] * scale[1]
    
    return xs, ys

def get_dist2(ax, ay, bx, by):
    return (ax - bx) ** 2 + (ay - by) ** 2

def get_dist(ax, ay, bx, by):
    return get_dist2(ax, ay, bx, by) ** 0.5
    
def get_ade(predicted_trajectory, actual_trajectory):
    assert(len(predicted_trajectory[0]) == len(predicted_trajectory[1]))
    assert(len(actual_trajectory[0]) == len(actual_trajectory[1]))
    
    assert(len(actual_trajectory[0]) == len(predicted_trajectory[0]))
    assert(len(actual_trajectory[1]) == len(predicted_trajectory[1]))
    
    sum_dist = 0
    total_points = 0
    
    # predicted = (px, py)
    # actual = (px, py)
    for ((px, py), (ax, ay)) in zip(zip(predicted_trajectory[0], predicted_trajectory[1]), zip(actual_trajectory[0], actual_trajectory[1])):
        dist = get_dist(px, py, ax, ay)
        
        sum_dist += dist
        total_points += 1
    
    ade = sum_dist / total_points
    
    return ade

def get_fde(predicted_trajectory, actual_trajectory):
    px, py = predicted_trajectory[0][-1], predicted_trajectory[1][-1]
    ax, ay = actual_trajectory[0][-1], actual_trajectory[1][-1]
    
    return get_dist(px, py, ax, ay)
    
def main(argc, argv):
    # torch.set_printoptions(sci_mode=False)
    
    # # get default device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # load dataset
    # dataset = VSPDatasetLoader(vsp_dir="data_zara", vsp_name="crowds_zara01.vsp", frame_size=(720, 576)).load()
    
    # # model settings
    # timesteps = 1000
    # in_steps = 50
    # out_steps = 100
    # num_agents = 5
    
    # # create diffusion model
    # ddpm = DDPM(timesteps, in_steps, out_steps, num_agents, device=device, noise_predictor="unet")
    
    # # create dataloader (manages batching and shuffling)
    # dataloader = VSPDataLoader(dataset, in_steps, out_steps, num_agents, batch_size=64)
    
    # # get a test sample
    # # TODO: I gotta jump through a lot of hoops just to get a single sample...
    # test_labels_all, test_conditioning_all = next(iter(dataloader))
    
    # test_labels = [test_labels_all[0]]
    # test_conditioning = [test_conditioning_all[0]]
    
    # test_labels, test_conditioning = format_batch((test_labels, test_conditioning), ddpm)
    
    # # run inference on test conditioning
    # ddpm.eval()
    
    # with torch.no_grad():
        # t = torch.tensor([1000]).to(device)
        
        # q_sample = ddpm.sample_q_distribution(test_labels, t)
        
        # print(test_labels)
        # print(q_sample)
        
        # predicted_trajectory = q_sample[0].reshape(-1, 2).cpu()
        # actual_trajectory = test_labels[0].reshape(-1, 2).cpu()

        # predicted_trajectory = list(predicted_trajectory)
        # actual_trajectory = list(actual_trajectory)
        
        # predicted_trajectory = HumanTrajectory([], predicted_trajectory)
        # actual_trajectory = HumanTrajectory([], actual_trajectory)
        
        # predicted_trajectory = predicted_trajectory.format_as_positions()
        # actual_trajectory = actual_trajectory.format_as_positions()
        
        # predicted_trajectory = np.array(predicted_trajectory.positions)
        # actual_trajectory = np.array(actual_trajectory.positions)
        
        # _, ax = plt.subplots()
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # ax.plot(actual_trajectory[:, 0], actual_trajectory[:, 1])
        
        # _, ax2 = plt.subplots()
        # ax2.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1])
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        
        # plt.show()
    
    
    #########################################
    
    torch.set_printoptions(sci_mode=False)
    
    # get default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load dataset
    data_frame_size = (720, 576)
    dataset = VSPDatasetLoader(vsp_dir="data_zara", vsp_name="crowds_zara01.vsp", frame_size=data_frame_size).load()
    
    # model settings
    timesteps = 1000
    in_steps = 20
    out_steps = 20
    num_agents = 5
    
    # create diffusion model
    ddpm = DDPM(timesteps, in_steps, out_steps, num_agents, device=device, noise_predictor="unet")
    
    # create dataloader (manages batching and shuffling)
    dataloader = VSPDataLoader(dataset, in_steps, out_steps, num_agents, batch_size=256)
    
    # train model
    ddpm.load_checkpoint("small_adv_unet_time_checkpoint.pth")
    # train(ddpm, dataloader, num_epochs=1000, learning_rate=1e-4)
    # ddpm.save_checkpoint("small_adv_unet_time_checkpoint.pth")
    
    # # get test metrics
    # test_sample_count = 100
    
    # test_dataset = VSPDatasetLoader(vsp_dir="data_zara", vsp_name="crowds_zara02.vsp", frame_size=data_frame_size).load()
    # test_dataloader = VSPDataLoader(test_dataset, in_steps, out_steps, num_agents, batch_size=test_sample_count)
    
    # test_labels, test_conditioning = get_next_formatted_batch(test_dataloader, ddpm)
    
    # # format samples
    # ddpm.eval()
    
    # with torch.no_grad():
        # ades = np.zeros(test_sample_count)
        # fdes = np.zeros(test_sample_count)
        
        # # run backward single on all test samples...
        # for i, (actual_trajectory, trajectory_conditioning) in enumerate(zip(test_labels, test_conditioning)):
            # initial_trajectory = test_conditioning[0][0]
            
            # predicted_trajectory = ddpm.backward_single(trajectory_conditioning)
            # predicted_trajectory = convert_tensor_trajectory_to_plottable(initial_trajectory, predicted_trajectory, include_initial=False, scale=data_frame_size)
            # actual_trajectory = convert_tensor_trajectory_to_plottable(initial_trajectory, actual_trajectory, include_initial=False, scale=data_frame_size)
            
            # single_ade = get_ade(predicted_trajectory, actual_trajectory)
            # single_fde = get_fde(predicted_trajectory, actual_trajectory)
            
            # ades[i] = single_ade
            # fdes[i] = single_fde
            
            # print(str(i+1) + "/" + str(test_sample_count))
        
        # print()
        
        # ade = np.mean(ades)
        # ade_std = np.std(ades)
        
        # fde = np.mean(fdes)
        # fde_std = np.std(fdes)
        
        # print("ADE: " + str(ade))
        # print("    stddev: " + str(ade_std))
        
        # print("FDE: " + str(fde))
        # print("    stddev: " + str(fde_std))
    
    # get a test sample
    # TODO: I gotta jump through a lot of hoops just to get a single sample...
    test_labels_all, test_conditioning_all = next(iter(dataloader))
    
    test_labels = [test_labels_all[0]]
    test_conditioning = [test_conditioning_all[0]]
    
    test_labels, test_conditioning = format_batch((test_labels, test_conditioning), ddpm)
    
    # run inference on test conditioning
    ddpm.eval()
    
    with torch.no_grad():
        initial_trajectory = test_conditioning[0][0]
        predicted_trajectory = ddpm.backward_single(test_conditioning.squeeze(0))
        
        initial = initial_trajectory
        
        # initial_trajectory = convert_tensor_trajectory_to_plottable(None, initial_trajectory)
        # predicted_trajectory = convert_tensor_trajectory_to_plottable(initial, predicted_trajectory)
        # actual_trajectory = convert_tensor_trajectory_to_plottable(initial, test_labels[0])
        
        initial_trajectory = convert_tensor_trajectory_to_plottable(None, initial_trajectory, scale=data_frame_size)
        predicted_trajectory = convert_tensor_trajectory_to_plottable(initial, predicted_trajectory, include_initial=False, scale=data_frame_size)
        actual_trajectory = convert_tensor_trajectory_to_plottable(initial, test_labels[0], include_initial=False, scale=data_frame_size)
        
        # get metrics for this particular test
        ade = get_ade(predicted_trajectory, actual_trajectory)
        fde = get_fde(predicted_trajectory, actual_trajectory)
        
        # print(predicted_trajectory)
        # print(actual_trajectory)
        
        xlim = -data_frame_size[0]/2, data_frame_size[0]/2
        ylim = -data_frame_size[1]/2, data_frame_size[1]/2
        
        _, ax = plt.subplots()
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        ax.plot(initial_trajectory[0], initial_trajectory[1], color="blue")
        ax.plot(actual_trajectory[0], actual_trajectory[1], color="green")
        ax.plot(predicted_trajectory[0], predicted_trajectory[1], color="red")
        
        # _, ax = plt.subplots()
        # plt.xlim(*xlim)
        # plt.ylim(*ylim)
        # ax.plot(actual_trajectory[0], actual_trajectory[1])
        
        # _, ax2 = plt.subplots()
        # ax2.plot(predicted_trajectory[0], predicted_trajectory[1])
        # plt.xlim(*xlim)
        # plt.ylim(*ylim)
        
        print("ADE: " + str(ade))
        print("FDE: " + str(fde))
        
        plt.show()
    
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)