from random import uniform
class neuron:
    def __init__(self,input):
        self.wi = [uniform(0,1) for i in range(0,len(input))]
        self.b = uniform(0,1)
        
