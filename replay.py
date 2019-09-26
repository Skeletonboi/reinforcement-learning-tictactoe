import random


class Replay():
    def __init__(self, size):
        self.exp = []
        self.pos = 0
        self.size = size
        self.firstpasscounter = 0

    def push(self, obj):
        if len(self.exp) < self.size:
            self.exp.append(None)
        self.exp[self.pos] = obj
        self.pos = (self.pos + 1) % self.size
        if self.firstpasscounter < self.size:
            self.firstpasscounter += 1

    def sample(self, sample_size):
        return random.sample(self.exp, sample_size)
