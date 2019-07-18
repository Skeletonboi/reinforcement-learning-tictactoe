from random import uniform
class neuron:
    def __init__(self,input_length):
        self.input_layer = [uniform(0,1) for i in range(0,input_length-1)]
        self.hidden_layers = [


#Q-Values for each action taken from a state estimated by neural network;
# Network input are states;
# Network output nodes are Q-values of each action taeken at the state;

# Objective is to learn optimal Q-function (state-action valuation function)
# that continues to satisfy Bellman optimality equation:
#q*(s,a) = E[R_t+1 + discount*max_a'(q(s',a')]

# NOTE: Stack of frames (in visual-input contexts) can be useful to gain more
# information (ie. speed, direction);

# Experiences
