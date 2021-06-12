from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import dataloader
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os, sys

import dcgan
import vae

import os, sys, random
from torchvision.transforms.transforms import Resize

from tqdm import tqdm

device_ = torch.device('cuda:0')

if __name__ == '__main__':
    # modelDG = torch.load('../toyDCGAN/model/model_saved_epoch100.pth')
    # D = dcgan.Discriminator().to(device_)
    # G = dcgan.Generator().to(device_)
    # D.load_state_dict(modelDG['D'])
    # G.load_state_dict(modelDG['G'])
    # for param in G.parameters():
    #     param.requires_grad = False
    Encoder = vae.VAEencoder().to(device_)
    Decoder = vae.VAEdecoder().to(device_)
    Encoder2 = vae.VAEencoder().to(device_)
    # for param in Encoder2.parameters():
    #     param.requires_grad = False

    dataset = dset.ImageFolder(
        root = '../toyDCGAN/minidata',
        transform=transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    totalnum = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    optE = optim.Adam(Encoder.parameters(), lr=1e-4)
    optD = optim.Adam(Decoder.parameters(), lr=1e-4)
    # G.eval()
    # D.eval()
    Encoder.train()
    state = torch.load('../toyVAE/model/vae_save_99_p2.pt')
    Encoder2.load_state_dict(state['encoder'])
    Encoder2.eval()
    for param in Encoder2.parameters():
        param.requires_grad = False
    for epoch in tqdm(range(100)):
        loss_ = 0.0
        temp = None
        # torch.save(Encoder.state_dict(), './encoder.temp')
        # state = torch.load('./encoder.temp')
        # Encoder2.load_state_dict(state)
        # for param in Encoder2.parameters():
        #     param.requires_grad = False
        # Encoder2.eval()
        f = open('./vae_enc_record.txt', 'a', encoding='utf8')
        for i, data in tqdm(enumerate(dataloader)):
            data = data[0].to(device_)
            batchnum = data.shape[0]
            optE.zero_grad()
            optD.zero_grad()
            z_mu, z_sigma = Encoder(data)
            noise = torch.randn(data.shape[0], vae.NZ, 1, 1).to(device_)
            z = z_mu + z_sigma * noise
            out_mu = Decoder(z)
            
            loss, l1, l2 = vae.encVAELoss(z_mu, z_sigma, out_mu, z, Encoder2, kl_weight=batchnum/totalnum)
            # loss, l1, l2 = vae.myVAELoss(z_mu, z_sigma, out_mu, data, kl_weight=batchnum/totalnum)
            loss_ += loss.item()
            print(loss.item(), l1.item(), l2.item())
            f.write('{} {} {}\n'.format(loss.item(), l1.item(), l2.item()))
            loss.backward()
            optE.step()
            optD.step()
            if (i + 1) % 100 == 0:
                print(loss_ / 100)
                loss_ = 0.0
            temp = out_mu.detach()
        vutils.save_image(temp / 2.0 + 0.5, './generated/vae_enc_{}.jpg'.format(epoch), nrow=8, normalize=True)
        f.close()
        torch.save({'E':Encoder, 'D':Decoder}, './model_enc_{}'.format(epoch))
