import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from game import tictactoe
class hyperparams():
    def __init__(self,lr,seed,numepoch):
        self.lr = lr
        self.seed = seed
        self.numepoch = numepoch

    def setLr(self,x):
        self.lr = x
        return self.lr

    def setSeed(self,x):
        self.seed = x
        return self.seed

    def setNumepoch(self,x):
        self.numepoch = x
        return self.numepoch


def accuracy(predictions, label):
    total_corr = 0
    index = 0
    for c in predictions.flatten():
        if (c.item() > 0.5):
            r = 1.0
        else:
            r = 0.0
        if (r == label[index].item()):
            total_corr += 1
        index += 1
    return (total_corr / len(label))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(9, 18)
        self.fc2 = nn.Linear(18, 18)
        self.fc3 = nn.Linear(18, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


