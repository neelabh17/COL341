import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sys

from skimage import io, transform

import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

from IPython.display import Image
import cv2

class BasicNNet(nn.Module):
    '''
    2.679178 M parameters
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 512, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(512, 1024, kernel_size=2, stride=1)

        self.fc1 = nn.Linear(1024, 256)

        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 10)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu(x)

        x = self.fc1(x.squeeze(2).squeeze(2))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        

        return x


class BB(nn.Module):
    
    def __init__(self, inp, out, reduction = True):
        super().__init__()
        self.c1 = nn.Conv2d(inp,inp,3,1, padding=1)
        self.c2 = nn.Conv2d(inp,inp,3,1, padding=1)
        self.c3 = nn.Conv2d(inp,inp,3,1, padding=1)
        self.bn1 = nn.BatchNorm2d(inp)
        self.bn2 = nn.BatchNorm2d(inp)
        self.bn3 = nn.BatchNorm2d(inp)
        self.bn4 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU()
        self.reduction = reduction
        if(self.reduction):
            self.c4 = nn.Conv2d(inp,out,3,2, padding=1)
            self.reducto = nn.Conv2d(inp,out,3,2, padding=1)
        else:
            self.c4 = nn.Conv2d(inp,inp,3,1, padding=1)


        
        
    
    def forward(self, x):
        _x = x
        _x = self.relu(self.bn1(self.c1(_x)))
        _x = self.relu(self.bn2(self.c2(_x)))

        x = x + _x
        _x = x
        _x = self.relu(self.bn3(self.c3(_x)))
        _x = self.relu(self.bn4(self.c4(_x)))
        
        if(self.reduction):
            x = _x + self.relu(self.reducto(x))
        else:
            x = x + _x

        return x

class BB_half(nn.Module):
    
    def __init__(self, inp, out, reduction = True):
        super().__init__()
        self.c3 = nn.Conv2d(inp,inp,3,1, padding=1)
        self.bn3 = nn.BatchNorm2d(inp)
        self.bn4 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU()
        self.reduction = reduction
        if(self.reduction):
            self.c4 = nn.Conv2d(inp,out,3,2, padding=1)
            self.reducto = nn.Conv2d(inp,out,3,2, padding=1)
        else:
            self.c4 = nn.Conv2d(inp,inp,3,1, padding=1)


        
        
    
    def forward(self, x):
        _x = x
        _x = self.relu(self.bn3(self.c3(_x)))
        _x = self.relu(self.bn4(self.c4(_x)))
        
        if(self.reduction):
            x = _x + self.relu(self.reducto(x))
        else:
            x = x + _x

        return x

class SmallResnet(nn.Module):
    '''
    ?? parameters
    '''


        
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, padding=3)
        self.mpool1 = nn.MaxPool2d(3,2, padding=1)
        self.b1 = BB(inp=64, out= 128, reduction=True)
        self.b2 = BB(inp=128, out= 256, reduction=True)
        self.conv2 = nn.Conv2d(256, 512, 3, 2, 1)
        self.relu = nn.ReLU()

        self.avg_pool = nn.AvgPool2d(7)

        self.fc1 = nn.Linear(512, 980)
        self.fc2 = nn.Linear(980, 10)
        self.dropout = nn.Dropout(0.2)
        
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.mpool1(x)
        x = self.b1(x)
        x = self.b2(x)
        # [14, 14, 256]
        x = self.relu(self.conv2(x))
        # [7, 7, 512]

        x = self.avg_pool(x)

        x = self.relu(self.fc1(x.squeeze(2).squeeze(2)))
        x = self.dropout(x)

        x = self.fc2(x)







        return x


def main():
    model = SmallResnet()
    print("Parmas = ", sum(p.numel() for p in model.parameters())/10**6, "M" )

    inp = torch.rand(200, 3, 224, 224)
    # trans = transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Resize((224,224))])
    print(inp.shape)
    # out = nn.Conv2d(3,65,3,2, padding=1)(inp)
    out = model(inp)
    print(out.shape)



    

if __name__ == "__main__":
    main()