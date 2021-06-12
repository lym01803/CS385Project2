import torch
import torch.nn as nn
import random
from torch.nn.init import xavier_normal_

from torch.nn.modules.activation import Tanh
from tqdm import tqdm

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

class lowDiscriminator(nn.Module):
    def __init__(self):
        super(lowDiscriminator, self).__init__()
        self.device = torch.device('cuda:0')
        self.fc = nn.Linear(3 * 64 * 64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.view(-1, 3 * 64 * 64)
        X = self.fc(X)
        X = self.sigmoid(X)
        return X

class RbfLR:
    def __init__(self, weight=10.0):
        self.num = None
        self.K = None
        self.X = None
        self.Y = None
        self.dim = None
        self.w = weight
        self.Alpha = None
        self.G = None
        self.v = None

    def fit(self, X, Y, max_epoch=500, Use_Adam=True, lr=1.0, gamma=0.9, beta=0.999, eps=1e-4):
        self.X = X
        self.Y = Y
        self.num = X.shape[0]
        self.dim = X.shape[1]
        self.Alpha = torch.zeros(self.num).cuda()
        self.Alpha.requires_grad = False
        self.K = torch.zeros(self.num, self.num).cuda()
        self.K.requires_grad = False
        batch_size = 64
        batch_num = (self.num - 1) // batch_size + 1
        idx = [i for i in range(self.num)]
        # idx = torch.tensor(idx, dtype=torch.long).cuda()
        for bi in tqdm(range(batch_num)):
            biidx = idx[bi*batch_size: bi*batch_size+batch_size]
            biidx = torch.tensor(biidx, dtype=torch.long).cuda()
            Xbi = torch.index_select(X, 0, biidx)
            for bj in range(batch_num):
                bjidx = idx[bj*batch_size: bj*batch_size+batch_size]
                bjidx = torch.tensor(bjidx, dtype=torch.long).cuda()
                Xbj = torch.index_select(X, 0, bjidx)
                X2i = torch.sum(Xbi * Xbi, dim=1)
                X2j = torch.sum(Xbj * Xbj, dim=1)
                Xij = torch.matmul(Xbi, Xbj.T)
                Dij = X2j - 2.0 * Xij + X2i.view(-1, 1)
                Dij = torch.exp(- Dij / self.w)
                idxi, idxj = torch.meshgrid(biidx, bjidx)
                self.K[idxi, idxj] = Dij
        self.v = torch.zeros(self.X.shape[0]).cuda()
        self.v.requires_grad = False
        self.G = torch.tensor(0.0).cuda()
        self.G.requires_grad = False
        for epoch in tqdm(range(max_epoch)):
            random.shuffle(idx)
            for b in range(batch_num):
                bidx = idx[b*batch_size: b*batch_size+batch_size]
                bidx = torch.tensor(bidx, dtype=torch.long).cuda()
                Yb = torch.index_select(Y, 0, bidx)
                Kb = torch.index_select(self.K, 0, bidx)
                Pb = torch.matmul(Kb, self.Alpha)
                Pb = torch.sigmoid(Pb)
                g = torch.matmul(Kb.T, Yb - Pb)
                if Use_Adam:
                    self.v = gamma * self.v + (1.0 - gamma) * g 
                    self.G = beta * self.G + (1.0 - beta) * torch.sum(g * g)
                    g = (self.v / (1.0 - gamma)) / torch.sqrt(self.G / (1.0 - beta) + eps)
                self.Alpha += lr * g
                if epoch % 100 == 0:
                    lr *= 0.1
    
    def predict(self, Xt):
        X = self.X
        n = self.num
        nt = Xt.shape[0]
        '''
        idx = [i for i in range(n)]
        idxt = [i for i in range(nt)]
        batch_size = 64
        batch_num = (n - 1) // batch_size + 1
        batch_numt = (nt - 1) // batch_size + 1
        TempK = torch.zeros((nt, n)).cuda()
        for bt in tqdm(range(batch_numt)):
            btidx = idxt[bt*batch_size: bt*batch_size+batch_size]
            btidx = torch.tensor(btidx, dtype=torch.long).cuda()
            Xbt = torch.index_select(Xt, 0, btidx)
            for b in range(batch_num):
                bidx = idx[b*batch_size: b*batch_size+batch_size]
                bidx = torch.tensor(bidx, dtype=torch.long).cuda()
                Xb = torch.index_select(X, 0, bidx)
                X2t = torch.sum(Xbt * Xbt, dim=1)
                X2 = torch.sum(Xb * Xb, dim=1)
                Xij = torch.matmul(Xbt, Xb.T)
                Dij = X2 - 2.0 * Xij + X2t.view(-1, 1)
                Dij = torch.exp(- Dij / self.w)
                idxbt, idxb = torch.meshgrid(btidx, bidx)
                TempK[idxbt, idxb] = Dij
        '''
        X2t = torch.sum(Xt * Xt, dim=1)
        X2 = torch.sum(X * X, dim=1)
        Xij = torch.matmul(Xt, X.T)
        Dij = X2 - 2.0 * Xij + X2t.view(-1, 1)
        Dij = torch.exp(- Dij / self.w)
        P = torch.matmul(Dij, self.Alpha)
        P = torch.sigmoid(P)
        return P.view(-1)
