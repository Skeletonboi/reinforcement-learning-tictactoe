import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from nn import MLP, hyperparams
from game import tictactoe
from replay import Replay
import math

# Tictactoe board: [x1,...,x8] vector of length=9
# 0 = empty
# 1 = self
# 2 = enemy
# Action: [0,...,1,...,0] vector of length=9
# all elements are binary
# at most one 1


# Experience Replay Initialization
replay_num = 20
m = Replay(replay_num)

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
numep = 10
# Agent can make at MOST 5 actions before forced termination
numt = 5
epsilon = 0.90

# Episode Loop
for ep in range(numep):
    print('new game')
    # Initialize tictactoe game
    x = tictactoe()
    terminate = False
    while not terminate:
        board = torch.from_numpy(x.outputState())
        rand = random.random()
        legal_rand = False
        if rand < epsilon:  # Random Action
            print('Random Action:')
            # Ensure it is legal, else resample
            while not legal_rand:
                a = random.randint(0, 8)
                if board[a].item() == 0:
                    legal_rand = True
        else:  # Exploit Action
            print('Exploit Action:')
            # Compute Q values of all actions
            predict = Qnet(board)
            q_max = torch.max(predict, 0)
            # Take highest Q value action
            a = q_max[1].item()
            # Ensure legality, else take next highest
            if board[a].item() == 0:
                legal_rand = True
            while not legal_rand:
                predict = torch.cat([predict[:a], predict[(a+1):]])
                q_max = torch.max(predict, 0)
                a = q_max[1].item()
                if board[a].item() == 0:
                    legal_rand = True

        # Execute Action
        print('a:',a)
        print('board b4:',board)
        x.move(2, math.floor(a / 3), a % 3)
        # Get new state
        board_next = torch.from_numpy(x.outputState())
        print('board after:',board_next)
        # Check and assign immediate state reward, rt
        stat = x.isGameOver()
        terminate = stat[0]
        if stat[0] == True:
            if stat[1] == 1:
                r = -5
            elif stat[1] == 2:
                r = 5
        elif stat[0] == False:
            r = -1
        # Store experience tuple in replay memory
        exp_t = tuple([board,a,r,board_next])
        m.push(exp_t)
        # Sample random minibatch of transitions(st,at,rt,st+1)
        exp_sample = m.sample(min(m.firstpasscounter,5))
        # Compute expected q(a|s_t) to optimize towards
        # (error from all possible actions are averaged to optimize against)
        for i in range(len(exp_sample)):
            temp = i[3]  # Extract board_t+1 tensor
            game_sample = tictactoe()
            for j in range(9):
                game_sample.state = [[i[3][state]]]
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
