import sys
import torch
import numpy as np
from crowds_data import VSPDatasetLoader, VSPDataset

def main(argc, argv):
    # load dataset
    data = VSPDatasetLoader(vsp_dir="data_zara", vsp_name="crowds_zara01.vsp", frame_size=(720, 576)).load()
    
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)