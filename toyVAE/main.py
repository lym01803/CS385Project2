import os, sys, random
import torch
import cv2
import numpy as np
import argparse

from torchvision.transforms.transforms import Resize
from vae import *
import torch
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

device_ = torch.device('cuda:0')
def train():
    
    dataset = dset.ImageFolder(
        root='../toyDCGAN/data',
        transform=transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    encoder = VAEencoder().to(device_)
    decoder = VAEdecoder().to(device_)
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    optE = optim.Adam(encoder.parameters(), lr=1e-6)
    optD = optim.Adam(decoder.parameters(), lr=1e-6)

    for epoch in tqdm(range(100)):
        loss_ = 0.0
        for i, data in tqdm(enumerate(dataloader)):
            data = data[0].to(device_)
            encoder.zero_grad()
            decoder.zero_grad()
            z_mu, z_sigma = encoder(data)
            noise = torch.randn(data.shape[0], NZ, 1, 1).to(device_)
            z = z_mu + z_sigma * noise
            out_mu, out_sigma = decoder(z)
            loss = myVAELoss(z_mu, z_sigma, out_mu, out_sigma, data)
            print(loss.item())
            loss_ += loss.item()
            loss.backward()
            optE.step()
            optD.step()
            if (i + 1) % 100 == 0:
                print(loss_ / (i + 1))
            if (i + 1) % 500 == 0:
                vutils.save_image(out_mu / 2.0 + 0.5, '../toyDCGAN/generated/vae_{}_{}.jpg'.format(epoch, i), nrow=8, normalize=True)

def main():
    manual_seed = 777
    random.seed(manual_seed)
    train()

if __name__ == '__main__':
    main()
