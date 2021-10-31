import torch.nn as nn
import torchvision.models as models
class model1(nn.Module):
    def __init__(self):
        super().__init__()

        print("INFO: Loaded Yoga Model")
        

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


