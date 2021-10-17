# from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# from skimage import io, transform

import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
# from tqdm import tqdm
# TODO remove TQDM dependencies

import cv2
import sys

# DataLoader Class
# if BATCH_SIZE = N, dataloader returns images tensor of size [N, C, H, W] and labels [N]
class ImageDataset(Dataset):
    
    def __init__(self, data_csv, train = True , img_transform=None):
        """
        Dataset init function
        
        INPUT:
        data_csv: Path to csv file containing [data, labels]
        train: 
            True: if the csv file has [labels,data] (Train data and Public Test Data) 
            False: if the csv file has only [data] and labels are not present.
        img_transform: List of preprocessing operations need to performed on image. 
        """
        
        self.data_csv = data_csv
        self.img_transform = img_transform
        self.is_train = train
        
        data = pd.read_csv(data_csv, header=None)
        if self.is_train:
            images = data.iloc[:,1:].to_numpy()
            labels = data.iloc[:,0].astype(int)
        else:
            images = data.iloc[:,:].to_numpy()
            labels = None
        
        self.images = images
        self.labels = labels
        # print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
    def __len__(self):
        """Returns total number of samples in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Loads image of the given index and performs preprocessing.
        
        INPUT: 
        idx: index of the image to be loaded.
        
        OUTPUT:
        sample: dictionary with keys images (Tensor of shape [1,C,H,W]) and labels (Tensor of labels [1]).
        """
        image = self.images[idx]
        image = np.array(image).astype(np.uint8).reshape((32, 32, 3), order = "F")
        
        if self.is_train:
            label = self.labels[idx]
        else:
            label = -1
        
        image = self.img_transform(image)
        
        sample = {"images": image, "labels": label}
        return sample


class NNet(nn.Module):
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




def main(args):
    test_file, model_file, pred_file = args[1:]
    # Data Loader Usage

    BATCH_SIZE = 200 # Batch Size. Adjust accordingly
    NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.

    img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])


    # Test DataLoader
    test_data = test_file # Path to test csv file
    test_dataset = ImageDataset(data_csv = test_data, train=False, img_transform=img_transforms)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

    model = NNet()
    model.load_state_dict(torch.load(model_file))

    if(torch.cuda.is_available()):
        model = model.cuda()

    # print("Parmas = ", sum(p.numel() for p in model.parameters())/10**6, "M" )
    _, preds = eval(model, test_loader, give_pred=True)
    with open(pred_file, "w") as f:
        for pred in preds.numpy().astype(np.int).tolist():
            f.write(str(pred)+"\n")

def eval(model, test_loader, give_pred = False):
    with torch.no_grad():
        if(give_pred):
            tot_pred = torch.Tensor([])
        else:
            tot_pred = None
        model.eval()
        correct = 0
        tots = 0
        # for sample in tqdm(test_loader, desc="Evaluating"):
        for sample in test_loader:
            images = sample['images']
            labels = sample['labels']

            if(torch.cuda.is_available()):
                images = images.cuda()
                labels = labels.cuda()

            preds = model(images).argmax(dim = 1)
            correct += (preds == labels).sum().item()
            tots += labels.shape[0]
            if(give_pred):
                tot_pred = torch.cat((tot_pred, preds.cpu()), dim = 0)

        
    return correct/tots, tot_pred
    # return correct/tots, tot_pred.numpy().astype(np.int).tolist()




if __name__ == "__main__":
    args = sys.argv
    main(args)