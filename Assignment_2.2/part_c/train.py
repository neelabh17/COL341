import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from skimage import io, transform

import matplotlib.pyplot as plt # for plotting
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

from IPython.display import Image
import cv2

from models import BasicNNet

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
        print("Total Images: {}, Data Shape = {}".format(len(self.images), images.shape))
        
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

def main(args):
    train_file, test_file, model_file, loss_file, acc_file, run_name = args[1:]
    # Data Loader Usage

    writer = SummaryWriter(comment=run_name)

    BATCH_SIZE = 200 # Batch Size. Adjust accordingly
    NUM_WORKERS = 20 # Number of threads to be used for image loading. Adjust accordingly.
    EPOCH = 50
    LR = 1e-4

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
    model = BasicNNet()

    print("Parmas = ", sum(p.numel() for p in model.parameters())/10**6, "M" )
    optim = torch.optim.Adam(model.parameters(), lr = LR)
    scheduler = StepLR(optim, step_size=30, gamma=0.1)

    if(torch.cuda.is_available()):
        model = model.cuda()
        # optim.cuda()
        criterion = criterion.cuda()


    avg_losses = []
    test_acc = []
    for ep in range(EPOCH):
        
        # Enumeration for 1 epoch
        total_loss = 0.0
        model.train()
        for sample in tqdm(train_loader, desc="Training Epoch:{}".format(ep+1)):
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
        scheduler.step()
        avg_loss = total_loss/len(train_loader)
        avg_losses.append(avg_loss)
        acc, _ = eval(model, test_loader)
        # acc_tr, _ = eval(model, train_loader)
        test_acc.append(acc)
        writer.add_scalar("Test Accuracy", acc, ep)
        writer.add_scalar("Train Loss", avg_loss, ep)
        # writer.flush()
        
        # with open(acc_file, "w") as f:
        #     for accu in test_acc:
        #         f.write(str(accu)+"\n")

        # plt.clf()
        # plt.rcParams["font.family"] = "serif"
        # plt.plot( [i+1 for i in range(len(test_acc))], test_acc)
        # plt.title("Test Accuracy vs Epoch")
        # plt.xlabel("Epoch")
        # plt.ylabel("Test Accuracy")
        # plt.savefig("test_acc.png")

        # with open(loss_file, "w") as f:
        #     for avg_l in avg_losses:
        #         f.write(str(avg_l)+"\n")
        # plt.clf()
        # plt.rcParams["font.family"] = "serif"
        # plt.plot([i+1 for i in range(len(avg_losses))], avg_losses)
        # plt.title("Train loss vs Epoch")
        # plt.xlabel("Epoch")
        # plt.ylabel("Train Loss")
        # plt.savefig("train_loss.png")
        # torch.save(model.state_dict(), model_file)

        print(total_loss/len(train_loader), acc)
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
        for sample in tqdm(test_loader, desc="Evaluating"):
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