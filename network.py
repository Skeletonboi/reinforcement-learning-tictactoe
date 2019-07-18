from random import uniform
class neuron:
    def __init__(self,input_length):
        self.input_layer = [uniform(0,1) for i in range(0,input_length-1)]
        self.hidden_layers = [


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
