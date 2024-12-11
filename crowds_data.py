# module for loading and parsing the crowds dataset

import os
from scipy.interpolate import CubicSpline

# utility class for managing VSP datasets
class VSPDataset():
    def __init__(self, trajectories):
        self.trajectories = trajectories
    

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
        trajectory = []
        
        # (including last frame)
        for frame in range(frames[0], frames[-1]+1):
            interp = tuple(cs(frame))
            
            trajectory.append((frame, interp))
        
        return trajectory

    # loads the contents of a vsp file as trajectories
    def load(logging=True):
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
                    norm_coord = self.normalize_point(pixel_x, pixel_y, frame_size)
                    
                    # add to spline
                    spline_frames.append(frame)
                    spline_points.append(norm_coord)
                
                # interpolate to a trajectory defined as frame-by-frame points
                trajectory = self.trajectory_spline_to_points(spline_frames, spline_points)
                
                trajectories.append(trajectory)
            
        if logging:
            print("Done.")
        
        return trajectories