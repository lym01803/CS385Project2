import os, sys, random
import torch
import cv2
import numpy as np
import argparse

from torchvision.transforms.transforms import Resize
# from vae import *
import torch
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from toyVAE.vae import *
from toyVAE import vae

import numpy as np

filepath = './toyDCGAN/data/list_attr_celeba.txt'

txt = np.loadtxt(filepath, skiprows=2, usecols=[i+1 for i in range(40)])
subtxt = txt[:25000]
count = 0
idx = []
label = []
for i in range(subtxt.shape[0]):
    if subtxt[i][35] == 1:
        idx.append(i)
        label.append(1)
    else:
        if random.random() < 0.05:
            idx.append(i)
            label.append(0)
print(len(label), sum(label))

Encoder = VAEencoder().cuda()
Decoder = VAEdecoder2().cuda()
save = torch.load('./toyVAE/model/vae_save_99_p2.pt')
Encoder.load_state_dict(save['encoder'])
Decoder.load_state_dict(save['decoder'])
dataset = dset.ImageFolder(
        root='./toyDCGAN/testdata',
        transform=transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
'''
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
for i, data in tqdm(enumerate(dataloader)):
    data = data[0].cuda()
    vutils.save_image(data, './input_5_64.jpg', nrow=8, normalize=True)
    z_mu, z_sigma = Encoder(data)
    noise = torch.randn(data.shape[0], NZ, 1, 1).cuda()
    z = z_mu + z_sigma * noise
    # out_mu, out_sigma = Decoder(z)
    out_mu = Decoder(z)
    vutils.save_image(out_mu, './output_5_64.jpg', nrow=8, normalize=True)
    
    p = torch.zeros_like(z)
    for i in range(64):
        p[i] = (1.0 - i/64.0) * z[0] + (i/64.0) * z[-1]
    # out_2, _ = Decoder(p)
    out_2 = Decoder(p)
    vutils.save_image(out_2, './linear_5_64.jpg', nrow=8, normalize=True)
    
    break

'''

Z = torch.randn(64, 100, 1, 1).cuda()
out, _ = Decoder(Z)
vutils.save_image(out, './random64_p.jpg', nrow=8, normalize=True)
