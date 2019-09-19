import torch
import torch.nn as nn

class SNC(nn.Module):
    def __init__(self):
        super(SNC,self).__init__()
        self.fc1 = nn.Linear(9,1)

    def forward(self,I):
        x = self.fc1(I)
        return x

smallNN = SNC()
print(smallNN.fc1.weight)