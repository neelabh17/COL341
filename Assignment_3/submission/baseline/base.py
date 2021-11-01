import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch.optim as optim

import os
import sys
import time
import numpy as np
import pandas as pd
import argparse

from PIL import Image
from tqdm import tqdm

TRAIN = True

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


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def Logger(children, prefix='INFO'):
    print(f'[{prefix}] {children}')


class model1(nn.Module):
    def __init__(self):
        super().__init__()

        Logger("Loaded Yoga Model 1")
        # print("INFO: Loaded Yoga Model 1")

        self.baseline = models.resnet50(pretrained=True)

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
            csv_path (string): path to csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        csv_path = path
        # if DEVELOPMENT:
        #     if(mode == "train"):
        #         csv_path = "data/training.csv"
        #     elif (mode == "val"):
        #         csv_path = "data/training.csv"
        #     elif (mode == "test"):
        #         csv_path = "data/test.csv"
        #     else:
        #         Logger("Wrong mode selected")
        #         # print("INFO: Wrong mode selected")
        # else:
        if mode != "train" and mode != "val" and mode != "test":
            Logger("Wrong mode selected")
            # print("INFO: Wrong mode selected")

        # TODO: FIX THE DATA_DIR THINGY
        # self.data_dir should be the folder location of the train.csv
        self.data_dir = "data"

        self.data = pd.read_csv(csv_path)
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
        img = Image.open(os.path.join(self.data_dir, self.image_loc[index]))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f'| {"Training" if TRAIN else "Testing"} | COL341 | Aryan Gupta | Neelabh Madan |')

    if TRAIN:
        parser.add_argument('--trainfolder', '-tf', default='.',
                            help='Path to training file')
        parser.add_argument('--outputfolder', '-of', default='.',
                            help='Path to save model weights to')
    else:
        parser.add_argument('--modelpath', '-mp', default='.',
                            help='Path to model weights folder')
        parser.add_argument('--testinput', '-ti', default='.',
                            help='Path to testfile')
        parser.add_argument('--testoutput', '-to', default='.',
                            help='Path to testoutput')

    args = parser.parse_args()

    Logger("Preparing Data")
    # print('==> Preparing data..')

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

    transform_test = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainloader = None
    valloader = None
    testloader = None

    if TRAIN:
        trainset = YogaDataset(args.trainfolder, "train", transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=64, shuffle=True, num_workers=8)

        valset = YogaDataset(args.trainfolder, "val", transform_val)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=64, shuffle=True, num_workers=8)
    else:
        testset = YogaDataset(args.testinput, "test", transform_train)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=8)

    net = model1()
    net = net.to(device)

    if not TRAIN:
        Logger("Loading Model")

        checkpoint = torch.load(args.modelpath)

        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

        Logger("Beginning test")

        net.eval()
        locs = []
        preds = []
        with torch.no_grad():
            for batch_idx, (inputs, _, img_loc) in enumerate(tqdm(testloader)):
                out = net(inputs.to(device))
                out = out.argmax(dim=1)
                out = out.squeeze(0)
                out = int(out.item())

                locs.append(img_loc)
                preds.append(class_list[out])

        locs = locs[:-1]
        preds = preds[:-1]

        locs = np.array(locs).reshape(-1, 1)
        preds = np.array(preds).reshape(-1, 1)
        data = np.concatenate((locs, preds), axis=1)

        df = pd.DataFrame(data, columns=["name", "category"])
        df.to_csv(args.testoutput, index=False)

    else:
        EPOCHS = 25
        Logger("Building Model")
        # print('==> Building model..')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200)

        # Training

        def train(epoch):
            Logger('\nEpoch: %d' % epoch)
            # print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets, _) in enumerate(trainloader):
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

                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        def test(epoch):
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets, _) in enumerate(valloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # Save checkpoint.
            acc = 100.*correct/total
            if acc > best_acc:
                Logger("Saving...")
                # print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }

                torch.save(state, args.outputfolder)
                best_acc = acc

        for epoch in range(start_epoch, start_epoch+EPOCHS):
            train(epoch)
            test(epoch)
            scheduler.step()
