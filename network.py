import random as rd
import numpy as np
class neuron:
    def __init__(self,type,input_len):
        self.weights = np.array(rd.uniform(0,1))
        # last unit of weights vec will be the bias
        self.output_value = 0
        self.type = type

    def update(self,input):
        # Neuron feeds-forward using the input data from previous input_layer
        # Input should be array data-type
        # Neuron output value depends on activation type (Relu or Sigmoid)
        input_b = input.append(1)
        value = np.dot(self.weights,input_b)
        if type == 'Relu':
            if value > 0:
                self.output_value = value
            else:
                self.output_value = 0
        elif type == 'sigmoid':
            self.output_value = 1/(1+np.exp(-value))


class neuralnet:
    def __init__(self,input_len,num_hidden,output_len):
        self.input_layer = [0 for i in range(0,input_length-1)]
        self.hidden_layers = []


#Q-Values for each action taken from each state estimated by a single neural net;
# Network input are states;
# Network output nodes are Q-values of each action taeken at the state;

# Objective is to learn optimal Q-function (state-action valuation function)
# that continues to satisfy Bellman optimality equation:
#q*(s,a) = E[R_t+1 + discount*max_a'(q(s',a')]

# Loss function will be calculated through the right hand side of Bellman optimality
# equation;

# NOTE: Stack of frames (in visual-input contexts) can be useful to gain more
# information (ie. speed, direction);

# Experience e_t as tuple of: e_t = (s_t,a_t,r_t+1,s_t+1);
# Last N# experiences per timestep per episode stored in replay memory;
# Want to randomly sample these experiences from replay memory to train network
# to prevent sequential correlation;

# s fed in to approximate q(s,a_1),q(s,a_2)...q(s,a_n);
# To calculate cost func, we need RHS of Bellman, which includes max_a'(q(s',a'))
# term. This term can be calculated by plugging s' (from exp. tuple) into the
# policy net and choosing max output q-value;

# Perform gradient descent using this cost func;
# BUT this doesn't quite make sense as you're "chasing your own tail"
# First-pass is optimized to be second-pass, but both passes are calculated using
# the same weights, therefore, SGDing weights for policy net will change your
# target value, a.k.a. constantly changing target value. => instablity;

# Solution: Separate "target DNN". Cloned NN from policy NN, but only updates its
# weights to the policy NN's weights periodically (period = hyperparameter).;

# Now first pass is in policy NN, second-pass (to find max-q term) is in target NN.;
