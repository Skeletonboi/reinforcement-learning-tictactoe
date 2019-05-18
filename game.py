import math as math
class tictactoe:
    def __init__(self):
        self.state = [[0,0,0],
                [0,0,0],
                [0,0,0]]
# 0 = empty, 1 = X, 2 = O

    def isGameOver(self):
        win = False
        for i in len(self.state):
            if self.state[i][0] == self.state[i][1] == self.state[i][2] != 0:
                win = True
        for i in len(self.state[0]):
            if self.state[0][i] == self.state[1][i] == self.state[2][i] != 0:
                win = True
        if self.state[0][0] == self.state[1][1] == self.state[2][2] != 0:
            win = True
        elif self.state[0][2] == self.state[1][1] == self.state[2][0] != 0:
            win = True
        return win

    def move(self,player,pos):
        row = math.floor(pos/3)
        if self.state[row][pos-3*row] != 0:
            print("Taken Spot!")
            return False
        elif pos > 8:
            print("Position Out of Bounds!")
            return False
        else:
            self.state[row][pos-3*row] == player
            return True

    def printState(self):
        a = self.state[0][0]
        b = self.state[0][1]
        c = self.state[0][2]
        d = self.state[1][0]
        e = self.state[1][1]
        f = self.state[1][2]
        g = self.state[2][0]
        h = self.state[2][1]
        i = self.state[2][2]
        print("{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}".format(a,b,c,d,e,f,g,h,i))
