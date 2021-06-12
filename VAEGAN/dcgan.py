import torch
import torch.nn as nn
import random

from torch.nn.modules.activation import Tanh

NC = 3
NZ = 100
NF = 64


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.device = torch.device('cuda:0')
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(NZ, NF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NF * 8),
            nn.ReLU(inplace=True), # 4 * 4
            nn.ConvTranspose2d(NF * 8, NF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 4),
            nn.ReLU(inplace=True), # 8 * 8
            nn.ConvTranspose2d(NF * 4, NF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 2),
            nn.ReLU(inplace=True), # 16 * 16
            nn.ConvTranspose2d(NF * 2, NF * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 1),
            nn.ReLU(inplace=True), # 32 * 32
            nn.ConvTranspose2d(NF * 1, NC, 4, 2, 1), # 64 * 64
            nn.Tanh()
        )
    
    def forward(self, X):
        return self.upsample(X)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.device = torch.device('cuda:0')
        self.downsample = nn.Sequential(
            nn.Conv2d(NC, NF * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 1),
            nn.ReLU(inplace=True), # 32
            nn.Conv2d(NF * 1, NF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 2),
            nn.ReLU(inplace=True), # 16
            nn.Conv2d(NF * 2, NF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 4),
            nn.ReLU(inplace=True), # 8,
            nn.Conv2d(NF * 4, NF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 8),
            nn.ReLU(inplace=True), # 512 * 4 * 4
            nn.Conv2d(NF * 8, 1, 4, 1, 0, bias=False), # 1 * 1 * 1
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.downsample(X)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
