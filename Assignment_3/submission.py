'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import os
import argparse

import numpy as np

from yoga_model import model1

from models import *
from utils import progress_bar

import pandas as pd

from dataloader import YogaDataset


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

# TODO: Removal: In this case only has the resume tag been hardcoded
parser.add_argument('--resume', '-r', action='store_true', default= "True",
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_val = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


testset = YogaDataset("test",transform_val)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=8)

classes = ['Virabhadrasana', 'Vrikshasana', 'Utkatasana', 'Padahastasana',
       'Katichakrasana', 'TriyakTadasana', 'Gorakshasana', 'Tadasana',
       'Pranamasana', 'ParivrittaTrikonasana', 'Tuladandasana',
       'Santolanasana', 'Still', 'Natavarasana', 'Garudasana',
       'Naukasana', 'Ardhachakrasana', 'Trikonasana', 'Natarajasana']

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = model1()
net = net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


net.eval()
locs = []
preds = []
with torch.no_grad():
    for batch_idx, (inputs, _, img_loc) in enumerate(tqdm(testloader)):
        out = net(inputs.cuda())
        out = out.argmax(dim = 1)
        out = out.squeeze(0)
        out = int(out.item())
        
        locs.append(img_loc)
        preds.append(classes[out])
        # import pdb; pdb.set_trace()

locs = locs[:-1]
preds = preds[:-1]

locs = np.array(locs).reshape(-1, 1)
preds = np.array(preds).reshape(-1, 1)
data = np.concatenate((locs, preds), axis = 1)

df = pd.DataFrame(data, columns = ["name","category"])
df.to_csv("test.csv", index = False)
# import pdb; pdb.set_trace()









