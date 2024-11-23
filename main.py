import sys
import torch
import numpy as np

from unet import UNet1d

def main(argc, argv):
    unet_test = UNet1d(1, 1)
    
    test_data = torch.tensor(np.random.rand(64).reshape(2, 1, 32), dtype=torch.float32)
    
    print(test_data)
    print(test_data.shape)
    
    test_result = unet_test.forward(test_data)
    
    print(test_result.shape)
    print(test_result)
    
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)