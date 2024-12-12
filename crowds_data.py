# module for loading and parsing the crowds dataset

import os
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset, DataLoader

class HumanTrajectory():
    def __init__(self, frames, positions):
        self.frames = frames
        self.positions = positions
    
    def __len__(self):
        return len(self.frames)
    
    def position_at_frame(self, frame):
        try:
            idx = self.frames.index(frame)
            
            return self.positions[idx]
        except:
            return None
        
    def unpack(self):
        return self.frames, self.positions
    
    def with_offset(self, offset):
        # re-map frames
        trajectory_frames = [frame + offset for frame in self.frames]
        
        return HumanTrajectory(trajectory_frames, self.positions)
    
    # remap frames to start at 0
    def as_root(self):
        initial = -self.frames[0]
        
        return self.with_offset(initial)
    
    # map the trajectory at source_index to the same frames as the target_index, with the assumption that the target_index's first frame is mapped to 0.
    def mapped_to_target(self, target):
        # get frame offset
        target_frames = target.frames
        
        offset = -target_frames[0]
        
        return self.with_offset(offset)
    
    # test if two trajectories overlap (occur within the same timeframe)
    def overlaps_with(self, target):
        target_frames = target.frames
        
        traj1_start, traj1_end = self.frames[0], self.frames[-1]
        traj2_start, traj2_end = target_frames[0], target_frames[-1]
        
        return not ( (traj1_end < traj2_start) or (traj1_start > traj2_end) )
    
    # clip trajectory by index
    def clip(self, start, end):
        return HumanTrajectory(self.frames[start:end], self.positions[start:end])
    
    # clip trajectory by frame (include every data point from start_frame to end_frame) and format to train on a model.
    # formatting means including a list of binary attributes that specify if a point is included in the trajectory.
    def clip_by_frame_and_format(self, start_frame, end_frame, exist_val=1, no_exist_val=0, no_exist_pos=(0, 0)):
        kept_frames = []
        kept_positions = []
        include_positions = []
        
        for frame in range(start_frame, end_frame):
            kept_frames.append(frame)
            
            if frame in self.frames:
                kept_positions.append(self.position_at_frame(frame))
                include_positions.append(exist_val)
            else:
                kept_positions.append(no_exist_pos)
                include_positions.append(no_exist_val)
        
        return HumanTrajectory(kept_frames, kept_positions), include_positions
        
    # returns two disjoint trajectories split along the split_index
    # first trajectory returned is everything before the split_index, second is everything after and including the split index
    def split(self, split_index):
        return HumanTrajectory(self.frames[:split_index], self.positions[:split_index]), HumanTrajectory(self.frames[split_index:], self.positions[split_index:])

def create_blank_trajectory(length):
    frames = [-1] * length
    positions = [(0, 0)] * length
    
    return HumanTrajectory(frames, positions)

# utility class for managing VSP datasets, mostly useful methods for training
class VSPDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories
    
    def __getitem__(self, i):
        return self.get_trajectory(i)
    
    def __len__(self):
        return len(self.trajectories)
        
    def num_trajectories(self):
        return len(self.trajectories)
    
    def get_trajectory_length_from_index(self, index):
        return len(self.get_trajectory(index))
    
    def get_trajectory(self, index):
        return self.trajectories[index]
    
    # get a single trajectory with frames adjusted to start at 0
    def get_trajectory_as_root(self, index):
        return self.get_trajectory(index).as_root()
    
    # see HumanTrajectory overlaps_with
    def do_trajectories_overlap(self, idx1, idx2):
        return self.get_trajectory(idx1).overlaps_with(self.get_trajectory(idx2))
    
    # see HumanTrajectory mapped_to_target
    def map_trajectory_to_target(self, source_index, target_index):
        return self.get_trajectory(source_index).mapped_to_target(self.get_trajectory(target_index))
    
    # gets the target index trajectory as a root trajectory as well as all other agents involved with that trajectory.
    # agent trajectories also have frames remapped to same space as root trajectory.  every frame before the "0" frame and after the last frame in the root trajectory is clipped.
    # an agent is counted as "involved" based on the return value of do_trajectories_overlap(root_trajectory, agent_trajectory)
    # the returned agents are also sorted in descending order by "influence", intended to be a simple metric of how much impact the agent has on the root trajectory.
    # in this case, influence is just the size of the trajectory after it's clipped.
    def get_trajectory_and_agents(self, index, sort_by_influence=True):
        root_trajectory = self.get_trajectory_as_root(index)
        
        root_trajectory_frames, root_trajectory_positions = root_trajectory.unpack()
        
        root_last_frame = root_trajectory_frames[-1]
        
        agent_trajectories = []
        
        # loop through all trajectories and find corresponding agents
        # TODO: there's probably a faster way to do this based on the fact that trajectories are in chronological order, but this is probably fine
        for agent_index in range(self.num_trajectories()):
            # don't test against self
            if agent_index == index:
                continue
            
            # test overlap
            if self.do_trajectories_overlap(index, agent_index):
                # get remapped agent trajectory
                remapped_agent_trajectory = self.map_trajectory_to_target(agent_index, index)
                
                # clip values
                agent_frames, agent_positions = remapped_agent_trajectory.unpack()
                
                clipped_frames = []
                clipped_positions = []
                
                for frame, position in zip(agent_frames, agent_positions):
                    if frame < 0 or frame > root_last_frame:
                        continue
                    
                    clipped_frames.append(frame)
                    clipped_positions.append(position)
                
                clipped_trajectory = HumanTrajectory(clipped_frames, clipped_positions)
                
                agent_trajectories.append(clipped_trajectory)
        
        if sort_by_influence:
            agent_trajectories.sort(reverse=True, key=lambda t : len(t))
        
        return root_trajectory, agent_trajectories
    
    def is_sample_valid(self, index, in_steps, out_steps, offset):
        root_trajectory = self.get_trajectory_as_root(index)
        
        total_steps = in_steps + out_steps
        
        # test if sample is complete given offset
        return len(root_trajectory)-offset >= total_steps
    
    # get a training sample for the model given a trajectory index, some number of timesteps, and an offset from the start of the trajectory.
    # training data consists of a trajectory over a fixed number of timesteps (no positions included) as well as the positions of all other agents in the scene at those timesteps.  for timesteps with no data, the position is 0, 0.  to help the model understand when a position is included or not, a binary attribute is attached to each position that's 1 if the position should be considered or 0 if the position isn't provided.  the training output is just the future trajectory positions, given by output_steps.  the binary attribute is not included, because it seems difficult for the model to learn to produce reasonable values (ie a list of ones followed by a list of zeros).  instead, we only train the model on samples where the output is a complete trajectory.
    # to get multiple training samples from a single long trajectory, an offset can be provided to modify where the training data begins from.  the function will raise an exception if the trajectory ends before steps+output_steps (we want to train the model only on complete samples).
    def get_training_sample(self, index, steps, output_steps, num_agents, offset=0):
        root_trajectory, agent_trajectories = self.get_trajectory_and_agents(index)
        
        # filter out agents we don't want
        agent_trajectories = agent_trajectories[:num_agents]
        
        total_steps = steps + output_steps
        
        # if there aren't enough agents, fill in the rest with blank trajectories
        # using negative frames for blank trajectories here is a rudimentary way to ensure that the trajectory isn't included
        if len(agent_trajectories) < num_agents:
            agent_trajectories = agent_trajectories + [create_blank_trajectory(total_steps)] * (num_agents - len(agent_trajectories))
        
        # test if sample is complete given offset
        if not self.is_sample_valid(index, steps, output_steps, offset):
            raise Exception("offset " + str(offset) + " too large to return a complete sample (needs " + str(total_steps) + " steps from trajectory of length " + str(len(root_trajectory)) + ")")
        
        # clip trajectories
        start = offset
        end = start + total_steps
        
        # we ignore include values for root, as root must be complete
        root_trajectory, root_include = root_trajectory.clip_by_frame_and_format(start, end)
        agent_trajectories = [traj.clip_by_frame_and_format(start, end) for traj in agent_trajectories]
        
        # split into input (data) and expected output (labels)
        root_data, root_label = root_trajectory.split(steps)
        root_include = root_include[:steps]
        
        agent_data = []
        agent_labels = []
        agent_includes = []
        
        for traj, include in agent_trajectories:
            traj_data, traj_label = traj.split(steps)
            include = include[:steps]
            
            agent_data.append(traj_data)
            agent_labels.append(traj_label)
            agent_includes.append(include)
        
        # agent labels are included but aren't expected to be used in training
        return (root_data, root_include, agent_data, agent_includes), (root_label, agent_labels)

# creates a thin wrapper around a torch DataLoader that facilitates proper fetching of training samples considering distinct offsets
# there's probably a more vanilla way to do this, but this works
class VSPDataLoader(Dataset):
    def __init__(self, vsp_dataset, in_steps, out_steps, num_agents, batch_size=64):
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.num_agents = num_agents
        self.vsp_dataset = vsp_dataset
        
        # get all possible index/offset combinations as list of tuples
        self.dataset = []
        
        for i, trajectory in enumerate( self.vsp_dataset):
            offset = 0
            
            while  self.vsp_dataset.is_sample_valid(i, self.in_steps, self.out_steps, offset):
                self.dataset.append((i, offset))
                offset += 1
        
        self.dataloader = DataLoader(self, batch_size=batch_size, shuffle=True)
    
    def __getitem__(self, i):
        return self.dataset[i]
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        trajectory_indices, offsets = next(iter(self.dataloader))
        
        labels = []
        conditioning = []
        
        for index, offset in zip(trajectory_indices, offsets):
            (root_data, root_include, agent_data, agent_includes), (root_label, agent_labels) = self.vsp_dataset.get_training_sample(index, self.in_steps, self.out_steps, self.num_agents, offset)
            
            labels.append(root_label)
            conditioning.append((root_data, agent_data, agent_includes))
        
        return labels, conditioning
    
    def get_next_batch(self):
        return next(iter(self))

# utility class for loading VSP datasets from a file
class VSPDatasetLoader():
    crowd_dataset_directory = "./datasets/crowd-data/crowds/data"
    
    def __init__(self, vsp_name, vsp_dir, frame_size):
        self.vsp_name = vsp_name
        self.vsp_dir = vsp_dir
        self.frame_size = frame_size
    
    # maps a point's coordinates to -1, 1 range
    def normalize_point(self, x, y):
        return (x / self.frame_size[0], y / self.frame_size[1])

    # given some control points for a spline, interpolate to obtain frame-by-frame coordinates from the start frame to the end frame
    # returns a list of coordinates in the following format:
    # (frame, (x, y))
    def trajectory_spline_to_points(self, frames, positions):
        cs = CubicSpline(frames, positions)
        
        # get trajectory points from spline_frames[0] to spline_frames[-1]
        trajectory_frames = []
        trajectory_positions = []
        
        # (including last frame)
        for frame in range(frames[0], frames[-1]+1):
            interp = tuple(cs(frame))
            
            trajectory_frames.append(frame)
            trajectory_positions.append(interp)
        
        return HumanTrajectory(trajectory_frames, trajectory_positions)

    # loads the contents of a vsp file as trajectories
    def load(self, logging=True):
        # TODO: this should probably use os.path but I'd have to use \\ for directories
        full_dir = self.crowd_dataset_directory + "/" + self.vsp_dir + "/" + self.vsp_name
        
        if not os.path.exists(full_dir):
            raise Exception("Bad filepath provided: " + str(full_dir))
        
        # read file
        with open(full_dir) as vsp_file:
            # get header data
            num_splines = int(vsp_file.readline().split(" ")[0])
            
            trajectories = []
            
            if logging:
                print("Loading " + str(num_splines) + " trajectories...")
            
            for i in range(num_splines):
                # get the number of data points
                num_points = int(vsp_file.readline().split(" ")[0])
                
                # build the trajectory spline
                spline_frames = []
                spline_points = []
                
                for p in range(num_points):
                    # read control point
                    control_point_line = vsp_file.readline()
                    
                    # parse items
                    control_point = control_point_line.split(" ")
                    
                    # parse to numbers
                    # NOTE: though it's provided in this dataset, we ignore gaze_direction for simplicity.  it would likely yield better results.
                    pixel_x, pixel_y, frame = control_point[:3]
                    
                    pixel_x, pixel_y = float(pixel_x), float(pixel_y)
                    frame = int(frame)
                    
                    # apply preprocessing
                    norm_coord = self.normalize_point(pixel_x, pixel_y)
                    
                    # add to spline
                    spline_frames.append(frame)
                    spline_points.append(norm_coord)
                
                # interpolate to a trajectory defined as frame-by-frame points
                trajectory = self.trajectory_spline_to_points(spline_frames, spline_points)
                
                trajectories.append(trajectory)
            
        if logging:
            print("Done.")
        
        return VSPDataset(trajectories)