import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sys

# from skimage import io, transform

import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
import glob
import os
# from tqdm import tqdm

# from IPython.display import Image
import cv2

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
    train_file, test_file, model_file, loss_file, acc_file = args[1:]
    # Data Loader Usage

    BATCH_SIZE = 200 # Batch Size. Adjust accordingly
    NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.
    EPOCH = 5

    img_transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

    # Train DataLoader
    train_data = train_file # Path to train csv file
    train_dataset = ImageDataset(data_csv = train_data, train=True, img_transform=img_transforms)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

    # Test DataLoader
    test_data = test_file # Path to test csv file
    test_dataset = ImageDataset(data_csv = test_data, train=True, img_transform=img_transforms)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)

    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(51)
    model = NNet()

    # print("Parmas = ", sum(p.numel() for p in model.parameters())/10**6, "M" )
    optim = torch.optim.Adam(model.parameters(), lr = 1e-4)

    if(torch.cuda.is_available()):
        model = model.cuda()
        # optim.cuda()
        criterion = criterion.cuda()


    avg_losses = []
    test_acc = []
    for _ in range(EPOCH):
        
        # Enumeration for 1 epoch
        total_loss = 0.0
        model.train()
        # for sample in tqdm(train_loader, desc="Training"):
        for sample in train_loader:
            images = sample['images']
            labels = sample['labels']

            if(torch.cuda.is_available()):
                images = images.cuda()
                labels = labels.cuda()

            preds = model(images)

            loss = criterion(preds, labels)
            total_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()


            # print(images.shape, labels.shape, model(images).shape)
        # print()
        avg_losses.append(total_loss/len(train_loader))
        acc, _ = eval(model, test_loader)
        # acc_tr, _ = eval(model, train_loader)
        test_acc.append(acc)
        
        with open(acc_file, "w") as f:
            for accu in test_acc:
                f.write(str(accu)+"\n")

        plt.clf()
        plt.rcParams["font.family"] = "serif"
        plt.plot( [i+1 for i in range(len(test_acc))], test_acc)
        plt.title("Test Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy")
        plt.savefig("test_acc.png")

        with open(loss_file, "w") as f:
            for avg_l in avg_losses:
                f.write(str(avg_l)+"\n")
        plt.clf()
        plt.rcParams["font.family"] = "serif"
        plt.plot([i+1 for i in range(len(avg_losses))], avg_losses)
        plt.title("Train loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.savefig("train_loss.png")
        torch.save(model.state_dict(), model_file)

        # print(total_loss/len(train_loader), acc)
        # print(total_loss/len(train_loader), acc, acc_tr)

        # print()

def eval(model, test_loader, give_pred = False):
    with torch.no_grad():
        if(give_pred):
            tot_pred = torch.Tensor([])
        else:
            tot_pred = None
        model.eval()
        correct = 0
        tots = 0
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