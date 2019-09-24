import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

# Tictactoe board: [x1,...,x8] vector of length=9
    # 0 = empty
    # 1 = self
    # 2 = enemy
# Action: [0,...,1,...,0] vector of length=9
    # all elements are binary
    # at most one 1


# Experience Replay Initialization
eN = 20
exp = [tuple([0,0,i]) for i in range(eN)]
print(exp[19][2])

# Initialize Q and target neural network
seed = 1
learningrate = 0.05
numepoch = 100
torch.manual_seed(seed)
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
        self.fc1 = nn.Linear(9, 15)
        self.fc2 = nn.Linear(15, 15)
        self.fc3 = nn.Linear(15, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

Qnet = MLP()
Tnet = MLP()

 # State Reward Function
