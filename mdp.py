import numpy as np
import torch
import random
from mrvic import neuron

input = [0 for i in range(0,9)]

hidden = [random.uniform(0,1) for i in range (0,5)]

output = [0 for i in range(0,2)]

for i in hidden:
    wi = [random.uniform(0,1) for i in range (0,len(input))]
    bi = random.uniform(0,1)
    yi = np.dot(wi,input)
    i = yi
