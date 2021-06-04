import torch
import torch.nn as nn
import random

from torch.nn.modules.activation import Tanh
from torch.nn.modules.batchnorm import BatchNorm2d

NC = 3
NZ = 100
NF = 64

'''
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
'''

class VAEencoder(nn.Module):
    def __init__(self):
        super(VAEencoder, self).__init__()
        self.device = torch.device('cuda:0')
        # 3 * 64 * 64
        self.feature_extract = nn.Sequential(
            nn.Conv2d(NC, NF * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 1),
            nn.ReLU(inplace=True), # NF * 32 * 32
            nn.Conv2d(NF * 1, NF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 2), # 128 * 16 * 16
            nn.ReLU(inplace=True),
            nn.Conv2d(NF * 2, NF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 4),
            nn.ReLU(inplace=True), # 256 * 8 * 8
            nn.Conv2d(NF * 4, NF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 8),
            nn.ReLU(inplace=True), # 512 * 4 * 4
        )
        self.fc_mu = nn.Conv2d(NF * 8, NZ, 4, 1, 0, bias=False)
        self.fc_sigma = nn.Conv2d(NF * 8, NZ, 4, 1, 0, bias= False)

    def forward(self, X):
        feature = self.feature_extract(X)
        mu = self.fc_mu(feature)
        sigma = self.fc_sigma(feature)
        return mu, sigma

class VAEdecoder(nn.Module):
    def __init__(self):
        super(VAEdecoder, self).__init__()
        self.device = torch.device('cuda:0')
        self.fc_mu = nn.ConvTranspose2d(NZ, NF * 8, 4, 1, 0, bias=False)
        self.fc_sigma = nn.ConvTranspose2d(NZ, NF * 8, 4, 1, 0, bias=False)
        self.output = nn.Sequential(
            nn.ConvTranspose2d(NF * 8, NF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(NF * 4, NF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NF * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(NF * 2, NF * 1, 4, 2, 1, bias = False),
            nn.BatchNorm2d(NF * 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(NF, NC, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, X):
        mu, sigma = self.fc_mu(X), self.fc_sigma(X)
        mu, sigma = self.output(mu), self.output(sigma)
        return mu, sigma

mseloss = nn.MSELoss()
def myVAELoss(mu_q, sigma_q, mu_p, sigma_p, X):
    global mseloss
    # mu_q, sigma_q : batch * 100 * 1 * 1
    # mu_p, sigma_p, X : batch * 3 * 64 * 64
    
    mu_q = mu_q.view(-1)
    sigma_q = sigma_q.view(-1)
    mu_q2 = mu_q ** 2
    sigma_q2 = sigma_q ** 2
    loss_1 = - torch.sum(torch.log(sigma_q2)) + torch.sum(sigma_q2) + torch.sum(mu_q2)
    
    mu_p = mu_p.view(-1)
    # sigma_p = sigma_p.view(-1)
    X_ = X.view(-1)
    # delta2 = (X_ - mu_p) ** 2
    # sigma_p2 = sigma_p ** 2
    # loss_2 = torch.sum(torch.log(sigma_p2)) + torch.sum(delta2 / sigma_p2)
    loss_2 = mseloss(X_, mu_p)
    
    loss = loss_1 + loss_2
    return loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
