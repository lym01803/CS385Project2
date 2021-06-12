import os, sys, random
import torch
import cv2
import numpy as np
import argparse
from torch._C import device

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

REPEAT = 0
REPEAT_D = 3

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
        root='./minidata',
        transform=transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    f = open('records_4v1.txt', 'a', encoding='utf8')
    f.write('-------------------------\n')
    noise_norm = 0.0
    def make_noise(batch, norm):
        noise = torch.randn(batch, 3, 64, 64).cuda()
        noise.requires_grad = False
        return noise * norm

    assert(REPEAT == 0 or REPEAT_D == 0)
    for epoch in tqdm(range(100)):
        modelD.train()
        modelG.train()
        Dloss = 0.0
        Gloss = 0.0
        for i, data in tqdm(enumerate(dataloader)):
            data = data[0].to(device_)
            modelD.zero_grad()
            logits = modelD(data + make_noise(data.shape[0], noise_norm)).view(-1)
            labels = torch.ones_like(logits)
            loss = criterion(logits, labels)
            loss.backward()
            Dloss += loss.mean().item()

            Z = torch.randn(data.shape[0], NZ, 1, 1).to(device_)
            GZ = modelG(Z)
            logits = modelD(GZ.detach() + make_noise(data.shape[0], noise_norm)).view(-1)
            labels = torch.zeros_like(logits)
            loss = criterion(logits, labels)
            loss.backward()
            Dloss += loss.mean().item()
            optD.step()
            
            if REPEAT > 0:
                for repeat in range(REPEAT):
                    Z = torch.randn(data.shape[0], NZ, 1, 1).to(device_)
                    Z.requires_grad = False
                    GZ = modelG(Z)
                    modelG.zero_grad()
                    logits = modelD(GZ + make_noise(data.shape[0], noise_norm)).view(-1)
                    labels = torch.ones_like(logits)
                    loss = criterion(logits, labels)
                    loss.backward()
                    Gloss += loss.mean().item()
                    optG.step()

            else:
                if (REPEAT_D == 0) or ((i + 1) % REPEAT_D == 0):
                    modelG.zero_grad()
                    logits = modelD(GZ + make_noise(data.shape[0], noise_norm)).view(-1)
                    labels = torch.ones_like(logits)
                    loss = criterion(logits, labels)
                    loss.backward()
                    Gloss += loss.mean().item()
                    optG.step()

            if (i + 1) % 100 == 0:
                # print(Dloss / (i + 1), Gloss / (i + 1))
                print(Dloss / 100, Gloss / (100 * (1 + REPEAT)))
                f.write('{} {}\n'.format(Dloss / 100, Gloss / (100 * (1 + REPEAT))))
                Dloss = 0.0
                Gloss = 0.0
        noise_norm = noise_norm * 0.8
        modelG.eval()
        with torch.no_grad():
            Gfixed = modelG(fixed_noise)
            vutils.save_image(Gfixed / 2.0 + 0.5, './generated2/generated_4v1_{}.jpg'.format(epoch), nrow=8, normalize=True)

        if (epoch + 1) % 10 == 0:
            to_save = {'D': modelD.state_dict(), 'G': modelG.state_dict()}
            torch.save(to_save, './model/4v1_model_saved_epoch{}.pth'.format(epoch + 1))

    f.close()
def main():
    manual_seed = 777
    random.seed(manual_seed)
    train()

if __name__ == '__main__':
    main()
