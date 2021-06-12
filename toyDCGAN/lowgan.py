import os, sys, random
import torch
import cv2
import numpy as np
import argparse
from torch.autograd.grad_mode import no_grad

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
    modelG = Generator().to(device_)
    modelD = lowDiscriminator().to(device_)
    dataset = dset.ImageFolder(
        root='./minidata',
        transform=transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    fixed_noise = torch.randn(64, NZ, 1, 1, device=device_)
    optG = optim.Adam(modelG.parameters(), lr=1e-4)
    optD = optim.Adam(modelD.parameters(), lr=3e-3)
    criterion = nn.BCELoss()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    '''
    for epoch in tqdm(range(100)):
        modelG.eval()
        Gloss = 0.0
        Dloss = 0.0
        # Train D first
        # prepare samples
        X = torch.zeros(12800, 3*64*64).cuda()
        X.requires_grad = False
        Y = torch.zeros(12800).cuda()
        Y.requires_grad = False
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                data = data[0]
                X[i*64: i*64+64] = data.view(64, -1)
                Y[i*64: i*64+64] = torch.ones(64)
                Z = torch.randn(data.shape[0], NZ, 1, 1).to(device_)
                GZ = modelG(Z)
                X[6400+i*64: 6400+i*64+64] = GZ.view(64, -1)
                if i + 1 == 100:
                    break
        # noise = torch.randn(12800, 3*64*64).cuda() / (1.0 + epoch**2) / 2
        modelD.fit(X, Y, max_epoch=100)
        modelG.train()
        f = open('records_low.txt', 'a', encoding='utf8')
        for i in tqdm(range(800)):
            modelG.zero_grad()
            Z = torch.randn(64, NZ, 1, 1).to(device_)
            GZ = modelG(Z)
            logits = modelD.predict(GZ.view(64, -1))
            # print(logits)
            labels = torch.ones_like(logits)
            loss = criterion(logits, labels)
            loss.backward()
            Gloss += loss.mean().item()
            optG.step()

            if (i + 1) % 100:
                print(Gloss / 100)
                f.write('{} {} {}\n'.format(epoch, i, Gloss / 1024)) 
                Gloss = 0.0

        modelG.eval()
        with torch.no_grad():
            Gfixed = modelG(fixed_noise)
            vutils.save_image(Gfixed / 2.0 + 0.5, './generated2/low_{}.jpg'.format(epoch), nrow=8, normalize=True)
        
        f.close()
        '''
    f = open('records_low.txt', 'a', encoding='utf8')
    f.write('-------------------------\n')
    for epoch in tqdm(range(100)):
        modelD.train()
        modelG.train()
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
                # print(Dloss / (i + 1), Gloss / (i + 1))
                print(Dloss / 100, Gloss / 100)
                f.write('{} {}\n'.format(Dloss / 100, Gloss / 100))
                Dloss = 0.0
                Gloss = 0.0

        modelG.eval()
        with torch.no_grad():
            Gfixed = modelG(fixed_noise)
            vutils.save_image(Gfixed / 2.0 + 0.5, './generated2/low_generated_{}.jpg'.format(epoch), nrow=8, normalize=True)

        if (epoch + 1) % 10 == 0:
            to_save = {'D': modelD.state_dict(), 'G': modelG.state_dict()}
            torch.save(to_save, './model/low_model_saved_epoch{}.pth'.format(epoch + 1))

if __name__ == '__main__':
    train()
