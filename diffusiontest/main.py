import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super().__init__()
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            torch.nn.BatchNorm2d(out_channels)
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            torch.nn.BatchNorm2d(out_channels)
        )
        
        if activation:
            self.activation = torch.nn.ReLU()
        else:
            self.activation = None
    
    def forward(self, x):
        x = self.conv1(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        # x = self.conv2(x)
        
        # if self.activation is not None:
            # x = self.activation(x)
        
        return x

# downsamples data to 1/2 and applies convolution    
class DownConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # first we downscale by a factor of 2,
        # then we apply a standard convolution to get output channels
        self.down_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2),
            Conv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.down_conv(x)

# upscales data by 2 and applies convolution, also applies skip connections
class UpConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # we have the transpose operation output half the input because the skip connections add the other half of the channels
        # then the convolution operation outputs the proper number of channels
        self.up_conv = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = Conv(in_channels, out_channels)
        
        # self.up_conv = torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        # self.conv = Conv(in_channels, out_channels)
    
    def forward(self, x, skip):
        # get upscaled layers
        x = self.up_conv(x)
        
        # concatenate with skip connection
        x = torch.cat([skip, x], dim=1)
        
        return self.conv(x)

class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# unet implementation
class UNet(torch.nn.Module):
    def __init__(self, input_channels, output_channels, activation="sigmoid"):
        super().__init__()
        
        # input layer converts input to feature layers
        self.inp = Conv(input_channels, 16)
        
        # downsamples data
        self.down1 = DownConv2d(16, 32)
        self.down2 = DownConv2d(32, 64)

        # upsamples data + apply skip connections
        self.up1 = UpConv2d(64, 32)
        self.up2 = UpConv2d(32, 16)
        
        # output layer converts feature layers to output channels
        self.out = torch.nn.Sequential(
            torch.nn.Conv2d(16, output_channels, kernel_size=3, padding="same"),
            # torch.nn.Tanh() if activation == "tanh" else torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.inp(x)
        
        d1 = self.down1(x)
        d2 = self.down2(d1)
        
        u1 = self.up1(d2, d1)
        u2 = self.up2(u1, x)
        
        return self.out(u2)
        
# class Denoiser(torch.nn.Module):
    # def __init__(self, num_channels):
        # super().__init__()
        
        # self.unet = UNet(num_channels, num_channels)
    
    # def forward(self, x):
        # return self.unet(x)
    
    # def inference_single(self, x):
        # with torch.no_grad():
            # self.eval()
            
            # x = torch.stack([x])
            
            # return self.forward(x)[0]

# define noise schedulers
def linear_noise_schedule(timesteps):
    b_min = 0.0001
    b_max = 0.02
    
    return torch.linspace(b_min, b_max, timesteps)

def get_alpha_values_and_schedule(noise_schedule):
    alpha_values = 1 - noise_schedule
    
    return alpha_values, torch.cumprod(alpha_values, axis=0)

class DiffusionModel():
    num_epochs = 5000
    batch_size = 64
    
    learning_rate = 3e-4
    
    def __init__(self, timesteps, shape):
        self.timesteps = timesteps
        
        self.noise_schedule = linear_noise_schedule(self.timesteps)
        self.alpha_values, self.alpha_schedule = get_alpha_values_and_schedule(self.noise_schedule)
        
        self.noise_predictor = UNet(2, 1)
        
        self.noise_predictor.to(device)
        
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.noise_predictor.parameters(), lr=self.learning_rate)
    
    def load_from_checkpoint(self, d):
        self.noise_predictor.load_state_dict(torch.load(d, weights_only=False))
        
    # embed timestep into input for single input/timestep pair
    def add_timestep_to_input_single(self, img, timestep):
        timestep_layer = torch.full(img.shape, (timestep / self.timesteps)*2 - 1 ).to(device)
        
        return torch.cat((img, timestep_layer))
    
    # embed timestep into input for batch of input/timestep pairs
    def add_timestep_to_input(self, imgs, timesteps):
        out = []
        
        for img, timestep in zip(imgs, timesteps):
            out.append(self.add_timestep_to_input_single(img, timestep))
        
        return torch.stack(out)
    
    # get diffused input x at timesteps t
    def forward_single(self, x, t):
        # sample random noise
        noise = torch.randn(x.shape).to(device)
        
        if t <= 0 or t > self.timesteps:
            return x, noise
        
        # apply noise to x
        alpha = self.alpha_schedule[t-1]
        
        diffused = torch.sqrt(alpha)*x + torch.sqrt(1 - alpha)*noise
        
        return diffused, noise
        
    def forward(self, x, t):
        forward_features = []
        forward_noises = []
        
        for x_i, t_i in zip(x, t):
            feature, noise = self.forward_single(x_i, t_i)
            
            forward_features.append(feature)
            forward_noises.append(noise)
        
        return torch.stack(forward_features), torch.stack(forward_noises)
    
    def backward_single(self, shape):
        # start from random noise
        self.noise_predictor.eval()
        
        with torch.no_grad():            
            current_image = torch.randn(shape).unsqueeze(1).to(device)
            
            for t in range(self.timesteps, 0, -1):
                alpha_val = self.alpha_values[t-1]
                alpha_hat = self.alpha_schedule[t-1]
                
                random_variance = torch.randn(current_image.shape).to(device)
                
                inp = self.add_timestep_to_input(current_image, torch.tensor([t]))
                
                predicted_noise = self.noise_predictor(inp)
                
                noise_coefficient = (1 - alpha_val) / math.sqrt(1 - alpha_hat)
                
                stddev = math.sqrt(self.noise_schedule[t-1])
                
                next_image = (1.0 / math.sqrt(alpha_val)) * (current_image - noise_coefficient*predicted_noise) + stddev*random_variance
                
                current_image = next_image
        
        return current_image[0]
        
    def get_random_timesteps(self, num):
        return torch.floor(torch.rand(num) * self.timesteps).int() + 1
        
    def train(self, train_data, save_dir):
        self.noise_predictor.train()
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        for i in range(self.num_epochs):
            total_loss = 0
            
            # # run mini-batches
            # for train_features, train_labels in train_loader:
                # train_features = train_features.to(device)
                # train_labels = train_labels.to(device)
            
                # diffused_images, actual_noise = self.forward(train_features, self.get_random_timesteps(len(train_features)))
                
                # # denoise images
                # predicted_noise = self.noise_predictor(diffused_images)
                
                # # get loss
                # total_loss += self.loss_func(predicted_noise, actual_noise)
            
            # run single mini-batch
            train_features, train_labels = next(iter(train_loader))
            
            train_features = train_features.to(device)
            train_labels = train_labels.to(device)
            
            # embed timesteps
            timesteps = self.get_random_timesteps(len(train_features))
            
            diffused_images, actual_noise = self.forward(train_features, timesteps)
            
            diffused_images_embedded = self.add_timestep_to_input(diffused_images, timesteps)
            
            # denoise images
            predicted_noise = self.noise_predictor(diffused_images_embedded)
            
            # get loss
            total_loss = self.loss_func(predicted_noise, actual_noise)
            
            # optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            if i % 50 == 0:
                print("Epoch:", i, " - Loss:", total_loss)
        
        
        torch.save(self.noise_predictor.state_dict(), save_dir)
        
        return
    

# def train_simple(model, img, inp):
    # model.to(device)
    # model.train()
    
    # inp = torch.stack([inp])
    # exp_out = torch.stack([img]).to(device)
    # inp = inp.to(device)
    
    # epochs = 1000
    # learning_rate = 1e-3
    
    # loss_func = torch.nn.MSELoss()
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # for i in range(epochs):
        # inp = torch.rand(img.shape)
        # inp = torch.stack([inp]).to(device)
        
        # model_out = model(inp)
        
        # loss = loss_func(model_out, exp_out)
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # if i % 10 == 0:
            # print("Epoch:", i, " - Loss:", loss)

# def train_less_simple(model, dataset, num_images):
    # model.to(device)
    # model.train()
    
    # # get images
    # train_images = torch.stack([dataset[i][0] for i in range(num_images)]).to(device)
    # inp = torch.stack([torch.rand(train_images[0].shape) for i in range(len(train_images))]).to(device)
    
    # epochs = 500
    # learning_rate = 1e-4
    
    # loss_func = torch.nn.MSELoss()
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # for i in range(epochs):
        # # produce input
        # # inp = torch.stack([torch.rand(train_images[0].shape) for i in range(len(train_images))]).to(device)
        
        # model_output = model(inp)
        
        # loss = loss_func(model_output, train_images)
        
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # if i % 10 == 0:
            # print("Epoch:", i, " - Loss:", loss)
    
def show_image_tensor(title, tensor, size, adjust=lambda x: x):
    tensor = adjust(tensor)
    
    img = tensor.permute(1, 2, 0).cpu().detach().numpy()
    
    old_height, old_width = img.shape[:2]
    
    new_height, new_width = old_height*size, old_width*size
    
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow(title, img)

tensor2img = torchvision.transforms.Lambda(lambda t: (t*0.5) + 0.5)
img2tensor = torchvision.transforms.Lambda(lambda t: (t*2) - 1)

def main(argc, argv):
    # map images to [-1, 1]
    train_dataset = MNIST("./MNIST/", train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        img2tensor
    ]))
    
    test_dataset = MNIST("./MNIST/", train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        img2tensor
    ]))
    
    # filter out only one class
    filter_label = 7
    
    filtered_dataset = []
    
    for img, label in test_dataset:
        if label == filter_label:
            filtered_dataset.append((img, label))
    
    timesteps = 1000
    
    diffusion = DiffusionModel(shape=train_dataset[0][0].shape, timesteps=timesteps)
    
    # diffusion.train(filtered_dataset, "test1.pth")
    diffusion.load_from_checkpoint("test_5000epochs_1000timesteps.pth")
    
    # test_image = filtered_dataset[0][0].to(device)
    
    # test_timestep = 200
    
    # diffused, noise = diffusion.forward_single(test_image, test_timestep)
    
    # predicted_noise = diffusion.noise_predictor(diffusion.add_timestep_to_input_single(diffused, test_timestep).unsqueeze(0))[0]
    
    # mse = torch.abs(predicted_noise - noise)
    
    # # get difference image
    # # diffused = torch.sqrt(alpha)*x + torch.sqrt(1 - alpha)*noise
    # # diffused - torch.sqrt(1 - alpha)*noise = torch.sqrt(alpha)*x
    # # (diffused - torch.sqrt(1 - alpha)*noise) / torch.sqrt(alpha) = x
    # alpha = diffusion.alpha_schedule[test_timestep-1]
    # rev = (diffused - torch.sqrt(1 - alpha)*predicted_noise) / torch.sqrt(alpha)
    
    # alpha_val = diffusion.alpha_values[test_timestep-1]
    # alpha_hat = diffusion.alpha_schedule[test_timestep-1]
    
    # show_image_tensor("original", test_image, 10, tensor2img)
    # show_image_tensor("diffused", diffused, 10, tensor2img)
    # show_image_tensor("noise", noise, 10, tensor2img)
    # show_image_tensor("predicted_noise", predicted_noise, 10, tensor2img)
    # show_image_tensor("mae", mse, 10)
    # show_image_tensor("reversed", rev, 10, tensor2img)
    
    # cv2.waitKey(0)
    
    for i in range(10):
        test = diffusion.backward_single(train_dataset[0][0].shape)
        
        # print(test)
        
        show_image_tensor("test " + str(i), test, 10, tensor2img)
        
    cv2.waitKey(0)
    
    # names = [str(i) for i in range(0, 110, 10)]
    # name = 0
    
    # for i in range(timesteps+1):
        # if i % (timesteps // 10) != 0:
            # continue
        
        # diffused, noise = diffusion.forward_single(test_dataset[3][0], i)
        
        # show_image_tensor(names[name], diffused, 10, tensor2img)
        # name += 1
    
    # cv2.waitKey(0)
    
    
    
# def main(argc, argv):
    # dataset = MNIST("./MNIST/", train=False, download=True, transform=torchvision.transforms.ToTensor())
    
    # denoiser = Denoiser(1)
    
    # test_image, test_label = dataset[0]
    
    # test_input = torch.rand(test_image.shape)
    
    # initial_image = denoiser.inference_single(test_input)
    
    # train_simple(denoiser, test_image, test_input)
    # # train_less_simple(denoiser, dataset, 100)
    
    # show_image_tensor("original", test_image, 10)
    # show_image_tensor("untrained", initial_image, 10)
    # show_image_tensor("trained", denoiser.inference_single(test_input.to(device)), 10)
    
    # while True:
        # random_input = torch.rand(test_image.shape).to(device)
        # random_output = denoiser.inference_single(random_input)
        
        # show_image_tensor("inp", random_input, 10)
        # show_image_tensor("random", random_output, 10)
        
        # if cv2.waitKey(1) == ord('q'):
            # break
            
    # # while True:
        # # random_input = torch.rand(test_image.shape).to(device)
        # # random_output = denoiser.inference_single(random_input)

        # # show_image_tensor("inp", random_input, 10)
        # # show_image_tensor("random", random_output, 10)

        # # if cv2.waitKey(1) == ord('q'):
            # # break

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)