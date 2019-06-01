import numpy as np
#import torch
import random
from neuron import neuron

input = [0 for i in range(0,9)]

hidden = [neuron(input) for i in range(0,5)]

output = [neuron(hidden) for i in range(0,2)]

cost = yowhat
