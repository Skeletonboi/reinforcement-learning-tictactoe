import math
import numpy as np
import torch

a = torch.tensor([1,2,3,4,5,6])
print(a)
a = torch.cat([a[:2],torch.tensor([0]), a[3:]])
print(a)
