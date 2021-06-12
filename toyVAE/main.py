import os, sys, random
# from toyVAE.vae import myVAELoss
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
        root='../toyDCGAN/minidata',
        transform=transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    totalnum = len(dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    encoder = VAEencoder().to(device_)
    decoder = VAEdecoder().to(device_)
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    optE = optim.Adam(encoder.parameters(), lr=1e-4)
    optD = optim.Adam(decoder.parameters(), lr=1e-4)

    for epoch in tqdm(range(100)):
        loss_ = 0.0
        f = open('./record_one.txt', 'a', encoding='utf8')
        for i, data in tqdm(enumerate(dataloader)):
            data = data[0].to(device_)
            batchnum = data.shape[0]
            encoder.zero_grad()
            decoder.zero_grad()
            z_mu, z_sigma = encoder(data)
            noise = torch.randn(data.shape[0], NZ, 1, 1).to(device_)
            z = z_mu + z_sigma * noise
            # out_mu, out_sigma = decoder(z)
            out_mu = decoder(z)
            # loss, l1, l2 = myVAELoss2(z_mu, z_sigma, out_mu, out_sigma, data, kl_weight=batchnum/totalnum)
            loss, l1, l2 = myVAELoss(z_mu, z_sigma, out_mu, data, kl_weight=batchnum/totalnum)
            # print(loss.item())
            loss_ += loss.item()
            print(loss.item(), l1.item(), l2.item())
            f.write('{} {} {}\n'.format(loss.item(), l1.item(), l2.item()))
            loss.backward()
            optE.step()
            optD.step()
            if (i + 1) % 100 == 0:
                print(loss_ / 100)
                loss_ = 0.0
            if (i + 1) % 300 == 0:
                vutils.save_image(out_mu / 2.0 + 0.5, '../toyDCGAN/generated/vae_{}_one.jpg'.format(epoch), nrow=8, normalize=True)
        # vutils.save_image(out_mu / 2.0 + 0.5, '../toyDCGAN/generated/vae_{}.jpg'.format(epoch), nrow=8, normalize=True)
        if  (epoch + 1) % 20 == 0:
            torch.save({'encoder':encoder.state_dict(), 'decoder':decoder.state_dict()}, './model/vae_save_{}_one.pt'.format(epoch))
        f.close()
def main():
    manual_seed = 777
    random.seed(manual_seed)
    train()

if __name__ == '__main__':
    main()
