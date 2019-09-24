import math as math
import pygame as pyg
import numpy as np

class tictactoe:
    def __init__(self):
        self.state = [[0,0,0],
                [0,0,0],
                [0,0,0]]
# 0 = empty, 1 = X, 2 = O

    def isGameOver(self):
        win = False
        for i in range(0,len(self.state)):
            if self.state[i][0] == self.state[i][1] == self.state[i][2] != 0:
                win = True
        for i in range(0,len(self.state[0])):
            if self.state[0][i] == self.state[1][i] == self.state[2][i] != 0:
                win = True
        if self.state[0][0] == self.state[1][1] == self.state[2][2] != 0:
            win = True
        elif self.state[0][2] == self.state[1][1] == self.state[2][0] != 0:
            win = True
        return win

    def reset(self):
        self.state = [[0,0,0],
                [0,0,0],
                [0,0,0]]
        return

    def move(self,player,row,col):

        if self.state[row][col] != 0:
            print("Taken Spot!")
            return False
        elif row > 2 | col > 2:
            print("Position Out of Bounds!")
            return False
        else:
            self.state[row][col] = player
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

    def drawState(self):
        pyg.display.init()
        display_height = 400
        display_width = 300
        disp = pyg.display.set_mode([display_width,display_height])
        disp.fill((255,255,255))
        pyg.draw.line(disp,(0,0,0),[100,0],[100,300],1)
        pyg.draw.line(disp,(0,0,0),[200,0],[200,300],1)
        pyg.draw.line(disp,(0,0,0),[0,100],[300,100],1)
        pyg.draw.line(disp,(0,0,0),[0,200],[300,200],1)

        clock = pyg.time.Clock()
        while True:
            mpos = pyg.mouse.get_pos()
            for i in pyg.event.get():
                if i.type == pyg.QUIT:
                    exit()
                elif i.type == pyg.KEYDOWN:
                    if i.key == 113:
                        exit()
                elif i.type == pyg.MOUSEBUTTONDOWN:
                    col = math.floor(mpos[0]/100)
                    row = math.floor(mpos[1]/100)
                    self.move(1,row,col)
                # Draw the game board
                for k in range(0,len(self.state)):
                    for l in range(0,len(self.state[k])):
                        if self.state[k][l] == 1:
                            #draw a X
                            self.printX(k,l,disp)
                        elif self.state[k][l] == 2:
                            #draw a O
                            self.printO(k,l,disp)



            pyg.display.update()
            if self.isGameOver() == True:
                pyg.time.delay(500)
                self.printClear(disp)
                self.reset()
            clock.tick(100)


    def printX(self,row,column,disp):
        pr = row*100 #Pixel Row
        pc = column*100 #Pixel Column
        pyg.draw.line(disp,(0,0,0),[pc+20,pr+20],[pc+80,pr+80],1)
        pyg.draw.line(disp,(0,0,0),[pc+80,pr+20],[pc+20,pr+80],1)
        return

    def printO(self,row,column,disp):
        pr = row*100 #Pixel Row
        pc = column*100 #Pixel Column
        pyg.draw.circle(disp,(0,0,0),[pc+50,pr+50],30,1)
        return

    def printClear(self,disp):
        disp.fill((255,255,255))
        pyg.draw.line(disp,(0,0,0),[100,0],[100,300],1)
        pyg.draw.line(disp,(0,0,0),[200,0],[200,300],1)
        pyg.draw.line(disp,(0,0,0),[0,100],[300,100],1)
        pyg.draw.line(disp,(0,0,0),[0,200],[300,200],1)
        return

    def outputState(self):
        out = []
        for i in range(len(self.state)):
            for j in range(3):
                out.append(self.state[i][j])
        return out



