import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from nn import MLP, hyperparams
from game import tictactoe
import math

# Tictactoe board: [x1,...,x8] vector of length=9
# 0 = empty
# 1 = self
# 2 = enemy
# Action: [0,...,1,...,0] vector of length=9
# all elements are binary
# at most one 1


# Experience Replay Initialization
eN = 20
exp = [tuple([0, 0, i]) for i in range(eN)]

# Initialize Q and target neural network
hyp = hyperparams(0.5, 2, 50)

torch.manual_seed(hyp.seed)


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


# Takes as input states (9 inputs), and outputs Q_value(a|s) for all possible a
Qnet = MLP()
# Target net is same as Qnet except is only updated every n runs
Tnet = MLP()

# MDP Hyperparams
numep = 50
numt = 5
epsilon = 0.90

# Episode Loop
for ep in range(numep):
    # Initialize tictactoe game
    x = tictactoe()
    board = x.outputState()
    board = torch.from_numpy(board)
    for t in range(numt):
        rand = random.random()
        legal_rand = False
        if rand < epsilon:  # Random Action
            # Ensure it is legal, else resample
            while not legal_rand:
                a = random.randint(0, 8)
                if board[a] == 0:
                    legal_rand = True
        else:  # Exploit Action
            # Compute Q values of all actions
            predict = Qnet(board)
            q_max = torch.max(predict, 0)
            # Take highest Q value action
            a = q_max[1].item()
            # Ensure legality, else take next highest
            while not legal_rand:
                if board[a] == 0:
                    legal_rand = True
                predict_new = torch.cat([predict[:a], torch.tensor([-999]),predict[a+1:]])
                q_max = torch.max(predict_new,0)
                a = q_max[1].item()

        # Execute Action
        legal = x.move(2, math.floor(a / 3), a % 3)
        # Get new state
        board_t = x.outputState()
        print(board_t)
        # Check and assign immediate state reward, rt
        temp = x.isGameOver()
        if temp[0] == True:
            if temp[1] == 1:
                rt = -5
            elif temp[1] == 2:
                rt = 5
        elif temp[0] == False:
            rt = -1
        # Store in experience tuple

#
#
# loss_function = nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=learningrate)
#
# lossRec = []
# vlossRec = []
# nRec = []
# trainAccRec = []
# validAccRec = []
# for i in range(numepoch):
#     optimizer.zero_grad()
#     predict = net(t_set)
#     loss = loss_function(input=predict.squeeze(), target=t_label.float())
#     loss.backward()
#     optimizer.step()
#     trainAcc = accuracy(predict, t_label)
#     # Computing Validation accuracy and loss
#     predict = net(v_set)
#     vloss = loss_function(input=predict.squeeze(), target=v_label.float())
#     validAcc = accuracy(predict, v_label)
#
#     lossRec.append(loss)
#     vlossRec.append(vloss)
#     nRec.append(i)
#     trainAccRec.append(trainAcc)
#     validAccRec.append(validAcc)
#
#  # State Reward Function
