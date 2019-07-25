import random as rd
import numpy as np


class neuron:
    def __init__(self,type,input_len):
        self.weights = []
        for i in range(input_len+1):
            self.weights.append(rd.uniform(0,1))
        # last unit of weights vec will be the bias
        self.output_value = rd.uniform(0,1)
        self.type = type

    def update(self,input):
        # Neuron feeds-forward using the input data from previous input_layer
        # Input should be array data-type
        # Neuron output value depends on activation type (Relu or Sigmoid)
        # Last item of weights vector is bias (for each neuron)
        input_b = input + [1]
        #print('this',input_b)
        #print('weight',self.weights)
        value = np.dot(self.weights,input_b)
        #print('that',value)
        if self.type == 'Relu':
            if value > 0:
                self.output_value = value
            else:
                self.output_value = 0
        elif self.type == 'sigmoid':
            self.output_value = 1/(1+np.exp(-value))

# TEMPORARY: NEURAL NET WILL BE HARDCODED AS RELU HIDDEN LAYERS AND SIGMOID OUTPUT LAYER.
class neuralnet:
    def __init__(self,input_len,hidden_len,hidden_num,output_len):
        self.input_len = input_len
        self.hidden_num = hidden_num
        self.hidden_len = hidden_len
        self.output_len = output_len

        self.input_layer = [-999 for i in range(0,input_len)]
        self.hidden_layers = [[neuron('Relu',input_len) for i in range(hidden_len)]]
        if hidden_num != 0:
            for k in range(hidden_num-1):
                self.hidden_layers.append([neuron('Relu',hidden_len) for i in range(hidden_len)])
                # Commented below is half-broken code for hidden layer initiliazation with varying hidden layer lengths
                #self.hidden_layers.append([neuron('Relu',len(self.hidden_layers[j])) for k in range(0,hidden_len)])
        self.output_layer = [neuron('sigmoid',hidden_len) for i in range(0,output_len)]



    def setInput(self,input_vec):
        # Ensure input_vector is the same size as the neural net input layer
        if len(input_vec) != len(self.input_layer):
            print('Attempted Input Update Vector size does not match network input size')
            return(False)
        # NEEDED: Ensures input_vector items are numerical (and within 0,1)

        # Update holding input_vector number
        self.input_layer = input_vec

    def forward_step(self):
        # Updates hidden layer neuron values with weight_vector multiplication and pass through to activation function
        # Starting with the first layer...
        for n in self.hidden_layers[0]:
            n.update(self.input_layer)

        # And every subsequent hidden layer...
        for i in range(1,self.hidden_num):
            prev_hid_vals = []
            for k in self.hidden_layers[i-1]:
                prev_hid_vals.append(k.output_value)
            for n in self.hidden_layers[i]:
                n.update(prev_hid_vals)
        # Updates output layer neuron values with "..."
        last_hid_vals = []
        for k in self.hidden_layers[-1]:
            last_hid_vals.append(k.output_value)
        for n in self.output_layer:
            n.update(last_hid_vals)



    def getCost(self,xis,yis):
        # xis is the input data vector, yi is/are the corresponding output value(s)
        self.setInput(xis)
        self.forward_step()
        # Getting output layer values
        y = []
        for i in self.output_layer:
            y.append(i.output_value)
        # RMSE Calculation
        return(np.sum(np.subtract(y,yis)**2)/self.output_len)


    def print_net(self):
        # Visualizes the neural network in an easy-to-understand fashion
        # Obtaining net layer sizes
        l_in = len(self.input_layer)
        l_hid_num = len(self.hidden_layers)
        l_hid = len(self.hidden_layers[0])
        l_out = len(self.output_layer)
        # Measuring visual parameters
        # Grid Dimensions:
        width = 2 + l_hid_num
        height = max(l_in,l_hid,l_out)

        g_in = (height - l_in)/2
        g_hid = (height - l_hid)/2
        g_out = (height - l_out)/2
        # Initializing empty net matrix
        net = [[0 for i in range(0,width)] for i in range(0,height)]
        # Writing the values of each Neuron
        for i in range(0,l_in):
            net[i][0] = self.input_layer[i]
        for j in range(0,l_hid_num):
            for i in range(0,l_hid):
                net[i][j+1] = self.hidden_layers[j][i].output_value
        for i in range(0,l_out):
            net[i][l_hid_num+1] = self.output_layer[i].output_value
        # Printing neural network as rows of arrays
        for k in range(0,len(net)):
            print(net[k])



x = neuralnet(4,5,2,2)
x.print_net()
inp_data = [[rd.uniform(0,1) for i in range(4)] for i in range(10)]
out_data = [[rd.uniform(0,1) for j in range(2)] for i in range (10)]

for i in range(10):
    print(x.getCost(inp_data[i],out_data[i]))




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
