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

device_ = torch.device('cuda:0')
modelG = Generator()
modelD = Discriminator()
modelG = modelG.to(device_)
modelG.apply(weights_init)
modelD = modelD.to(device_)
modelD.apply(weights_init)

gstate = torch.load('./model_saved_epoch100.pth')
modelG.load_state_dict(gstate['G'])

modelG.eval()
with torch.no_grad():
    Z = torch.randn(256, NZ, 1, 1).to(device_)
    for i in range(16):
        z1 = Z[i*16]
        z2 = Z[i*16+15]
        for j in range(16):
            Z[i*16+j] = z1 * (1.0-j/15.0) + z2 * (j/15.0)
    GZ = modelG(Z)
    vutils.save_image(GZ / 2.0 + 0.5, './show.tmp1.jpg', nrow=16, normalize=True)
