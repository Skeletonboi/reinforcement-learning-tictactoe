import torch
import torch.nn as nn
import torch.nn.Functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import game.py

# Tictactoe board: [x1,...,x8] vector of length=9
    # 0 = empty
    # 1 = self
    # 2 = enemy
# Action: [0,...,1,...,0] vector of length=9
    # all elements are binary
    # at most one 1


#Experience Replay Initialization