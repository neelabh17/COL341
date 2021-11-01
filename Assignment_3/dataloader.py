from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch

import os
import numpy as np
import pandas as pd

from PIL import Image
import torchvision
from tqdm import tqdm



class_list = ['Virabhadrasana', 'Vrikshasana', 'Utkatasana', 'Padahastasana',
       'Katichakrasana', 'TriyakTadasana', 'Gorakshasana', 'Tadasana',
       'Pranamasana', 'ParivrittaTrikonasana', 'Tuladandasana',
       'Santolanasana', 'Still', 'Natavarasana', 'Garudasana',
       'Naukasana', 'Ardhachakrasana', 'Trikonasana', 'Natarajasana']

class_name_to_id = {}
for i, class_name in enumerate(class_list):
    class_name_to_id[class_name] = i


# Taken from https://github.com/utkuozbulak/pytorch-custom-dataset-examples
class YogaDataset(Dataset):
    def __init__(self, mode = "train", transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        if(mode == "train"):
            csv_path = "data/training.csv"
        elif (mode =="val"):
            csv_path = "data/training.csv"
        elif (mode =="test"):
            csv_path = "data/test.csv"
        else:
            print("INFO: Wrong mode selected")


        self.data_dir = "data"
        self.data = pd.read_csv(csv_path)
        self.image_loc = np.asarray(self.data["name"])
        if(mode == "train" or mode == "val"):
            self.labels = np.asarray([class_name_to_id[class_name] for class_name in self.data["category"]])
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

        print("INFO: Data loaded")

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir,self.image_loc[index]))
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

    transform_train = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # yoga_dataset = \
    #     YogaDataset("test", transform_train)

    # print(len(yoga_dataset))
    # img, _ = yoga_dataset[0]
    # import pdb; pdb.set_trace()


    from yoga_model import model1
    # model = model1().cuda()
    model = torchvision.models.densenet201(pretrained=True).cuda()

    import pdb;pdb.set_trace()
    trainset = YogaDataset("train",transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=8, shuffle=True, num_workers=8)

    for (imgs, labels, _) in tqdm(trainloader):
        # print(imgs.shape)
        out = model(imgs.cuda())
        print(out.shape)

    

    # img = torch.rand(8,3,224, 224)
    # out = model(img)

    import pdb; pdb.set_trace()

