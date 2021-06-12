import numpy as np
from matplotlib import pyplot
import seaborn as sns
import pandas as pd

def plotGAN():
    filepath = './toyDCGAN/records.txt'
    txt = np.loadtxt(filepath)
    print(txt.shape)
    Gloss = np.zeros((100, ))
    Dloss = np.zeros((100, ))
    for i in range(100):
        # print(np.mean(txt[i*3: i*3+3, 0]))
        Dloss[i] = np.mean(txt[i*3: i*3+3, 0])
        Gloss[i] = np.mean(txt[i*3: i*3+3, 1])
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.plot(Dloss, label='Dloss')
    pyplot.plot(Gloss, label='Gloss')
    pyplot.legend()
    pyplot.show()
    pyplot.clf()
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.plot(Dloss, label='Dloss')
    pyplot.legend()
    pyplot.show()
    pyplot.clf()
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.plot(Gloss, label='Gloss')
    pyplot.legend()
    pyplot.show()
    pyplot.clf()

def plotGANNoise():
    filepath = './toyDCGAN/records.txt'
    txt = np.loadtxt(filepath)
    print(txt.shape)
    Gloss = np.zeros((100, ))
    Dloss = np.zeros((100, ))
    for i in range(100):
        # print(np.mean(txt[i*3: i*3+3, 0]))
        Dloss[i] = np.mean(txt[i*3: i*3+3, 0])
        Gloss[i] = np.mean(txt[i*3: i*3+3, 1])
    filepath = './toyDCGAN/records_4v1.txt'
    txt = np.loadtxt(filepath)
    print(txt.shape)
    Gloss2 = np.zeros((100, ))
    Dloss2 = np.zeros((100, ))
    for i in range(100):
        # print(np.mean(txt[i*3: i*3+3, 0]))
        Dloss2[i] = np.mean(txt[i*3: i*3+3, 0])
        Gloss2[i] = np.mean(txt[i*3: i*3+3, 1])
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.plot(Dloss, label='Dloss (GAN)')
    pyplot.plot(Dloss2, label='Dloss (GAN 4v1)')
    pyplot.legend()
    pyplot.show()
    pyplot.clf()
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.plot(Gloss, label='Gloss (GAN)')
    pyplot.plot(Gloss2, label='Gloss (GAN 4v1)')
    pyplot.legend()
    pyplot.show()
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.plot(Dloss, label='Dloss (GAN)')
    pyplot.plot(Gloss, label='Gloss (GAN)')
    pyplot.plot(Dloss2, label='Dloss (GAN4v1)')
    pyplot.plot(Gloss2, label='Gloss (GAN4v1)')
    pyplot.legend()
    pyplot.show()


plotGANNoise()
