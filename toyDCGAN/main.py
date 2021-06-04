import os, sys, random
import torch
import cv2
import numpy as np
import argparse

from torchvision.transforms.transforms import Resize
from dcgan import *
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
    modelG = Generator()
    modelD = Discriminator()
    modelG = modelG.to(device_)
    modelG.apply(weights_init)
    modelD = modelD.to(device_)
    modelD.apply(weights_init)
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, NZ, 1, 1, device=device_)
    optG = optim.Adam(modelG.parameters(), lr=1e-4)
    optD = optim.Adam(modelD.parameters(), lr=1e-4)
    
    dataset = dset.ImageFolder(
        root='./data',
        transform=transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in tqdm(range(100)):
        Dloss = 0.0
        Gloss = 0.0
        for i, data in tqdm(enumerate(dataloader)):
            data = data[0].to(device_)
            modelD.zero_grad()
            logits = modelD(data).view(-1)
            labels = torch.ones_like(logits)
            loss = criterion(logits, labels)
            loss.backward()
            Dloss += loss.mean().item()

            Z = torch.randn(data.shape[0], NZ, 1, 1).to(device_)
            GZ = modelG(Z)
            logits = modelD(GZ.detach()).view(-1)
            labels = torch.zeros_like(logits)
            loss = criterion(logits, labels)
            loss.backward()
            Dloss += loss.mean().item()
            optD.step()

            modelG.zero_grad()
            logits = modelD(GZ).view(-1)
            labels = torch.ones_like(logits)
            loss = criterion(logits, labels)
            loss.backward()
            Gloss += loss.mean().item()
            optG.step()

            if (i + 1) % 100 == 0:
                print(Dloss / (i + 1), Gloss / (i + 1))

            if (i + 1) % 500 == 0:
                vutils.save_image(GZ / 2.0 + 0.5, './generated/generated_{}_{}.jpg'.format(epoch, i), nrow=8, normalize=True)

        if (epoch + 1) % 10 == 0:
            to_save = {'D': modelD.state_dict(), 'G': modelG.state_dict()}
            torch.save(to_save, './model/model_saved.pth')

def main():
    manual_seed = 777
    random.seed(manual_seed)
    train()

if __name__ == '__main__':
    main()
