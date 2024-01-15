import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import numpy as np

import argparse
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, crop_size=60, transform=None):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name).convert("RGB")  # Convert to RGB if the image has an alpha channel

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = transforms.functional.crop(image, i, j, h, w)

        if self.transform:
            image = self.transform(image)
        return image
    


# Example usage:
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

dataset = CustomDataset(root_dir='C://Users//ANT//Documents//info//IIN_universal_inverse_problem//test_images//color//BSD100', crop_size=60, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Function to add Gaussian noise to images
def add_gaussian_noise(images, std_dev):
    noise = torch.randn_like(images) * std_dev
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)  # Ensure pixel values are within [0, 1]


class DenoiserAutoencoder(nn.Module):
    def __init__(self, bias = False):
        super(DenoiserAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=bias),  # Adjusted stride
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),  # Adjusted stride and output_padding
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias),
            nn.Sigmoid()  # Assuming images are normalized to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
##
def def_args(grayscale=False):
    '''
    @ grayscale: if True, number of input and output channels are set to 1. Otherwise 3
    @ training_data: models provided in here have been trained on {BSD400, mnist, BSD300}
    @ training_noise: standard deviation of noise during training the denoiser
    '''
    parser = argparse.ArgumentParser(description='BF_CNN_color')
    parser.add_argument('--dir_name', default= '../noise_range_')
    parser.add_argument('--kernel_size', default= 3)
    parser.add_argument('--padding', default= 1)
    parser.add_argument('--num_kernels', default= 64)
    parser.add_argument('--num_layers', default= 20)
    if grayscale is True: 
        parser.add_argument('--num_channels', default= 1)
    else:
        parser.add_argument('--num_channels', default= 3)
    
    args = parser.parse_args('')
    return args

class BF_CNN(nn.Module):

    def __init__(self, args):
        super(BF_CNN, self).__init__()

        self.padding = args.padding
        self.num_kernels = args.num_kernels
        self.kernel_size = args.kernel_size
        self.num_layers = args.num_layers
        self.num_channels = args.num_channels

        self.conv_layers = nn.ModuleList([])
        self.running_sd = nn.ParameterList([])
        self.gammas = nn.ParameterList([])


        self.conv_layers.append(nn.Conv2d(self.num_channels,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))

        for l in range(1,self.num_layers-1):
            self.conv_layers.append(nn.Conv2d(self.num_kernels ,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))
            self.running_sd.append( nn.Parameter(torch.ones(1,self.num_kernels,1,1), requires_grad=False) )
            g = (torch.randn( (1,self.num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
            self.gammas.append(nn.Parameter(g, requires_grad=True) )

        self.conv_layers.append(nn.Conv2d(self.num_kernels,self.num_channels, self.kernel_size, padding=self.padding , bias=False))


    def forward(self, x):
        relu = nn.ReLU(inplace=True)
        x = relu(self.conv_layers[0](x))
        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            # BF_BatchNorm
            sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ 1e-05)

            if self.conv_layers[l].training:
                x = x / sd_x.expand_as(x)
                self.running_sd[l-1].data = (1-.1) * self.running_sd[l-1].data + .1 * sd_x
                x = x * self.gammas[l-1].expand_as(x)

            else:
                x = x / self.running_sd[l-1].expand_as(x)
                x = x * self.gammas[l-1].expand_as(x)

            x = relu(x)

        x = self.conv_layers[-1](x)

        return x

class AdvancedDenoiser(nn.Module):
    def __init__(self, use_bias=True):
        super(AdvancedDenoiser, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


args = def_args()
# Initialize the model, loss function, and optimizer
model = AdvancedDenoiser(use_bias=False)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load your dataset (replace 'your_dataset_path' with your actual path)
# transform = transforms.Compose([transforms.ToTensor()])
# dataset = ImageFolder(root='C://Users//ANT//Documents//info//IIN_universal_inverse_problem//test_images//color//BSD100', transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Training loop
num_epochs = 50
max_std_dev = 1  # Set the initial standard deviation of Gaussian noise

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for data in dataloader:
        std_dev = np.random.uniform(0,max_std_dev)
        clean_images = data

        # Add Gaussian noise to create noisy images
        noisy_images = add_gaussian_noise(clean_images, std_dev)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(noisy_images)
        loss = criterion(outputs, clean_images)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'advanced_denoiser_without_biais.pth')

# Initialize the model, loss function, and optimizer
model = AdvancedDenoiser(use_bias=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load your dataset (replace 'your_dataset_path' with your actual path)
# transform = transforms.Compose([transforms.ToTensor()])
# dataset = ImageFolder(root='C://Users//ANT//Documents//info//IIN_universal_inverse_problem//test_images//color//BSD100', transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Training loop
num_epochs = 50
max_std_dev = 1  # Set the initial standard deviation of Gaussian noise

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for data in dataloader:
        std_dev = np.random.uniform(0,max_std_dev)
        clean_images = data

        # Add Gaussian noise to create noisy images
        noisy_images = add_gaussian_noise(clean_images, std_dev)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(noisy_images)
        loss = criterion(outputs, clean_images)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'advanced_denoiser_with_bias.pth')