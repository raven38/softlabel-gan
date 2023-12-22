# -*- coding: utf-8 -*-
"""
We need to create train and val folders manually before running the script 


@author: Dipu
"""

import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy
import imageio
import os

data_train = torchvision.datasets.STL10('./data', split='train', download=True,
                             transform=torchvision.transforms.Compose([
                             ]))
# data_train = torchvision.datasets.CIFAR10('./data', train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                              ]))
folderlocation = './data/cifar10/'

for iter1 in range(10):    # 10 = number of classes
    path = folderlocation + 'train/'+str(iter1)
    if not os.path.exists(path):
        os.makedirs(path)
    path = folderlocation + 'val/'+str(iter1)
    if not os.path.exists(path):
        os.makedirs(path)

for iter1 in range(len(data_train)):
    x, a = data_train[iter1] 
    imageio.imwrite(folderlocation + 'train/'+str(a)+'/train'+str(iter1)+'.png', x)
    
data_test = torchvision.datasets.STL10('./data', split='test', download=True,
                             transform=torchvision.transforms.Compose([
                             ]))
# data_test = torchvision.datasets.CIFAR10('./data', train=False, download=True,
#                              transform=torchvision.transforms.Compose([
#                              ]))
for iter1 in range(len(data_test)):
    x, a = data_test[iter1] 
    imageio.imwrite(folderlocation + 'val/'+str(a)+'/test'+str(iter1)+'.png', x)
