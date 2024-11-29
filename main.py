import sys
import torch
import numpy as np

from unet import UNet1d

def train_unet(unet, training_data, training_labels):
    epochs = 1000
    learning_rate = 1e-2
    loss_func = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    
    unet.train()
    
    for i in range(epochs):
        model_output = unet(training_data)
        
        loss = loss_func(model_output, training_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print("Epoch:", i, " - Loss:", loss)
    
def main(argc, argv):
    unet_test = UNet1d(1, 1)
    
    # generate made up input data
    samples = 100
    input_data = torch.tensor(np.random.rand(samples, 1, 16)).float()
    output_data = input_data + 1
    
    print("Training...")
    train_unet(unet_test, input_data, output_data)
    
    test_sample = 0
    
    unet_test.eval()
    
    with torch.no_grad():
        print("\nTesting...")
        
        print("in:", input_data)
        print("test:",unet_test(input_data))
        print("exp.:",output_data)
    
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)