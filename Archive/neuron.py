import random as rd
import numpy as np
# Neuron class for neural network object
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
