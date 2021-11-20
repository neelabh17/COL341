import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch.optim as optim
from efficientnet_pytorch import EfficientNet

import os
import sys
import time
import numpy as np
import pandas as pd
import argparse

from PIL import Image
from tqdm import tqdm

TRAIN = False
IN_KAGGLE = False


NEELARYA_MODEL_NAME = "model_2018ME10698_2018ME0672.pth"

class_list = ['Virabhadrasana', 'Vrikshasana', 'Utkatasana', 'Padahastasana',
              'Katichakrasana', 'TriyakTadasana', 'Gorakshasana', 'Tadasana',
              'Pranamasana', 'ParivrittaTrikonasana', 'Tuladandasana',
              'Santolanasana', 'Still', 'Natavarasana', 'Garudasana',
              'Naukasana', 'Ardhachakrasana', 'Trikonasana', 'Natarajasana']

class_name_to_id = {}
for i, class_name in enumerate(class_list):
    class_name_to_id[class_name] = i

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

MODELS = {
    "MODEL_1": 'model_1'
}

MODEL = MODELS["MODEL_1"]

CONFIG = {
    MODELS["MODEL_1"]: {
        "verbose": "Proper Readable Model Name",
        "TRAIN": True,
        "DEVELOPMENT": True,
        "NUM_WORKERS": 4,
        "TRAIN_PARAMS": {
            "BATCH_SIZE": 32,
            "SHUFFLE": True,
            "EPOCHS": 6,
            "LEARNING_RATE": 0.001,
            "MOMENTUM": 0.9,
            "WEIGHT_DECAY": (5e-4),
            "T_MAX": 200,
            "TRANSFORMATIONS": transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]),
        },
        "VAL_PARAMS": {
            "BATCH_SIZE": 32,
            "SHUFFLE": True,
            "TRANSFORMATIONS": transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        },
        "TEST_PARAMS": {
            "BATCH_SIZE": 32,
            "SHUFFLE": False,
            "TRANSFORMATIONS": transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        }
    }
}


def Logger(children, prefix='INFO'):
    print(f'[{prefix}] {children}')


class model1(nn.Module):
    def __init__(self):
        super().__init__()

        Logger("Loaded Yoga Model 1")
        # print("INFO: Loaded Yoga Model 1")


        self.baseline = EfficientNet.from_pretrained('efficientnet-b3')

        self.extra_fc1 = nn.Linear(1000, 512)
        self.extra_fc2 = nn.Linear(512, 128)
        self.extra_fc3 = nn.Linear(128, 19)
        self.relu = nn.ReLU()

    def forward(self, X):

        X = self.baseline(X)

        X = self.relu(self.extra_fc1(X))
        X = self.relu(self.extra_fc2(X))
        X = self.extra_fc3(X)

        return X


# Taken from https://github.com/utkuozbulak/pytorch-custom-dataset-examples
class YogaDataset(Dataset):
    def __init__(self, path="", mode="train", transforms=None):
        """
        Args:
            self.csv_path (string): path to csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.csv_path = path
        # if DEVELOPMENT:
        #     if(mode == "train"):
        #         self.csv_path = "data/training.csv"
        #     elif (mode == "val"):
        #         self.csv_path = "data/training.csv"
        #     elif (mode == "test"):
        #         self.csv_path = "data/test.csv"
        #     else:
        #         Logger("Wrong mode selected")
        #         # print("INFO: Wrong mode selected")
        # else:
        if mode != "train" and mode != "val" and mode != "test":
            Logger("Wrong mode selected")
            # print("INFO: Wrong mode selected")

        self.data_dir = os.path.dirname(self.csv_path)

        self.data = pd.read_csv(self.csv_path)
        self.image_loc = np.asarray(self.data["name"])
        if(mode == "train" or mode == "val"):
            self.labels = np.asarray(
                [class_name_to_id[class_name] for class_name in self.data["category"]])
        else:
            self.labels = [-1 for _ in range(len(self.image_loc))]

        if(mode == "train"):
            self.labels = self.labels[:int(len(self.labels)*0.8)]
            self.image_loc = self.image_loc[:int(len(self.image_loc)*0.8)]
        if(mode == "val"):
            self.labels = self.labels[int(len(self.labels)*0.8):]
            self.image_loc = self.image_loc[int(len(self.image_loc)*0.8):]

        assert len(self.labels) == len(self.image_loc)
        # import pdb; pdb.set_trace()

        self.transforms = transforms

        Logger("Data loaded")
        # print("INFO: Data loaded")

    def __getitem__(self, index):
        img = Image.open(os.path.join(
            self.data_dir, self.image_loc[index][2:]))
        # Transform image to tensor
        if self.transforms is not None:
            img = self.transforms(img)
        # Return image and the label

        label = self.labels[index]
        img_loc = self.image_loc[index]
        # Will be None for test set
        return (img, label, img_loc)

    def __len__(self):
        return len(self.labels)
        # return 8


if __name__ == "__main__":

    args = {}

    if not IN_KAGGLE:
        parser = argparse.ArgumentParser(
            description=f'| {"Training" if TRAIN else "Testing"} | COL341 | Aryan Gupta | Neelabh Madan |')

        if TRAIN:
            parser.add_argument('--traininput', '-tf', default='.',
                                help='Path to training file')
            parser.add_argument('--trainoutput', '-of', default='.',
                                help='Path to save model weights to')
        else:
            parser.add_argument('--modelpath', '-mp', default='.',
                                help='Path to model weights folder')
            parser.add_argument('--testinput', '-ti', default='.',
                                help='Path to testfile')
            parser.add_argument('--testoutput', '-to', default='.',
                                help='Path to testoutput')

        args_alias = parser.parse_args()
        if TRAIN:
            args = {
                "traininput": args_alias.traininput ,
                "trainoutput": args_alias.trainoutput, 
            }
        else:
            args = {
                "modelpath": args_alias.modelpath, 
                "testinput": args_alias.testinput ,
                "testoutput": args_alias.testoutput ,
            }
    else:
        if TRAIN:
            args = {
                "traininput": "/kaggle/input/col341-a3/training.csv",
                "trainoutput": "/kaggle/working",
            }
        else:
            args = {
                "modelpath": "/kaggle/working",
                "testinput": "/kaggle/input/col341-a3/test.csv",
                "testoutput": "/kaggle/working/submission.csv",
            }

    Logger("Preparing Data")
    # print('==> Preparing data..')

    transform_train = CONFIG[MODEL]['TRAIN_PARAMS']['TRANSFORMATIONS']

    transform_val = CONFIG[MODEL]['VAL_PARAMS']['TRANSFORMATIONS']

    transform_test = CONFIG[MODEL]['TEST_PARAMS']['TRANSFORMATIONS']

    trainloader = None
    valloader = None
    testloader = None

    if TRAIN:
        trainset = YogaDataset(args["traininput"], "train", transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=CONFIG[MODEL]['TRAIN_PARAMS']['BATCH_SIZE'], shuffle=CONFIG[MODEL]['TRAIN_PARAMS']['SHUFFLE'], num_workers=CONFIG[MODEL]['NUM_WORKERS'])

        valset = YogaDataset(args["traininput"], "val", transform_val)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=CONFIG[MODEL]['VAL_PARAMS']['BATCH_SIZE'], shuffle=CONFIG[MODEL]['VAL_PARAMS']['SHUFFLE'], num_workers=CONFIG[MODEL]['NUM_WORKERS'])
    else:
        testset = YogaDataset(args["testinput"], "test", transform_train)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=CONFIG[MODEL]['TEST_PARAMS']['BATCH_SIZE'], shuffle=CONFIG[MODEL]['TEST_PARAMS']['SHUFFLE'], num_workers=CONFIG[MODEL]['NUM_WORKERS'])

    net = model1()
    net = net.to(device)

    if not TRAIN:
        Logger("Loading Model")

        checkpoint = torch.load(os.path.join(args["modelpath"], NEELARYA_MODEL_NAME))

        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

        Logger("Beginning test")

        net.eval()
        locs = np.array([]).reshape(-1,1)
        preds = np.array([]).reshape(-1,1)
        with torch.no_grad():
            for batch_idx, (inputs, _, img_loc) in enumerate(tqdm(testloader)):
                out = net(inputs.to(device))
                out = out.argmax(dim=1)
                out = out.squeeze(0)
                
                img_loc_new = np.array(img_loc).reshape(-1,1)

                locs = np.concatenate((locs, img_loc_new), axis = 0)
                preds = np.concatenate((preds, out.cpu().numpy().reshape(-1, 1)), axis = 0)

        final_preds = []
        for i in range(preds.shape[0]):
            final_preds.append(class_list[int(preds[i,:])])
        preds = final_preds
        locs = locs[:-1, :]
        preds = preds[:-1]

        preds = np.array(preds).reshape(-1, 1)
        data = np.concatenate((locs, preds), axis=1)

        df = pd.DataFrame(data, columns=["name", "category"])
        df.to_csv(args["testoutput"], index=False)

    else:
        # Training

        EPOCHS = CONFIG[MODEL]['TRAIN_PARAMS']['EPOCHS']
        Logger("Building Model")
        # print('==> Building model..')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=CONFIG[MODEL]['TRAIN_PARAMS']['LEARNING_RATE'],
                              momentum=CONFIG[MODEL]['TRAIN_PARAMS']['MOMENTUM'], weight_decay=CONFIG[MODEL]['TRAIN_PARAMS']['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG[MODEL]['TRAIN_PARAMS']['T_MAX'])

        def train(epoch):
            Logger('\nEpoch: %d' % epoch)
            # print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            pbar = tqdm(trainloader, desc="Training")
            for batch_idx, (inputs, targets, _) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pbar.set_postfix_str('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        def test(epoch):
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(valloader, desc="Validation")
                for batch_idx, (inputs, targets, _) in enumerate(pbar):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    pbar.set_postfix_str('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # Save checkpoint.
            acc = 100.*correct/total
            if True:
                # We want to save on every iteration
                # if acc > best_acc:
                Logger("Saving...")
                # print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                # TODO: CONFIRM ABOUT THE SLASH HERE
                torch.save(
                    state, os.path.join(args["trainoutput"],NEELARYA_MODEL_NAME))
                best_acc = acc
                
                torch.save(
                    state, os.path.join(args["trainoutput"], str(epoch) + "_" + NEELARYA_MODEL_NAME))
                best_acc = acc

        for epoch in range(start_epoch, start_epoch+EPOCHS):
            train(epoch)
            test(epoch)
            scheduler.step()
