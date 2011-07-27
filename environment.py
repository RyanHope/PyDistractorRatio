__author__ = 'Ryan M. Hope <rmh3093@gmail.com>'

from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment
from helpers import *

class DistractorRatio(Environment, Named):
    
    objects = None
    fixation_location = (0,0)
    curAction = -1
    prevAction = 0
    keyPressed = 0
    targetVisible = 0
    nfix = 1
    ratio = 0
    random = False
    done = False

    def __init__(self, **args):#ratio, target_type, **args):
        self.setArgs(**args)
        #self.ratio = ratio
        #self.target_type = target_type
        self.space = np.linspace(-7.75,7.75,8)
        self.x,self.y = np.meshgrid(self.space,self.space)

    def reset(self):
        self.done = False
        self.nfix = 1
        self.targetVisible = 0
        self.keyPressed = 0
        self.fixation_location = (0,0)
        if self.random:
            self.ratio = choice(range(0,15))
            print '~~~~~~~~~~~~~~~ Testing Ratio %d ~~~~~~~~~~~~~~~~~~~~~' % (self.ratio+1)
        else:
            self.ratio = (self.ratio+1)%15
            print '~~~~~~~~~~~~~~~ Training Ratio %d ~~~~~~~~~~~~~~~~~~~~~' % (self.ratio+1)
        self.objects = generateObjects(self.ratio+1, 1)
        self.objects = apply_availability(self.objects,self.fixation_location[0],self.fixation_location[1])
        self.objects = score(self.objects)

    def performAction(self, action):
        self.prevAction = self.curAction
        self.curAction = action
        if action == 65:
            self.keyPressed = 1
        else:
            newFix = index2fix(action)
            self.fixation_location = newFix
            self.objects = apply_availability(self.objects,self.fixation_location[0],self.fixation_location[1])
            self.objects = score(self.objects)
            if targetVisible(self.objects):
                self.targetVisible = 1
            self.nfix = self.nfix + 1

    def getSensors(self):
        #eye = 8*find_nearest(self.space, self.fixation_location[1])[0] + find_nearest(self.space, self.fixation_location[0])[0]
        map = pmap(self.x,self.y,self.objects,score=9) * pmap(self.x,self.y,self.objects,score=8)
        return np.append(map.flatten(), self.targetVisible) 
        #return encodeState(pmap(self.x,self.y,self.objects,score=9),self.fixation_location)

    def getObservation(self):
        return self.getSensors()

    def getReward(self):
        reward = 0.0
        if self.targetVisible:
            reward = reward - self.nfix
        if self.curAction == self.prevAction:
            reward = reward - 1.0
        if self.nfix == 49:
            reward = reward - 1000
            self.done = True
        if self.keyPressed and self.targetVisible:
            reward = reward + 1000
        return reward
        
    def isFinished(self):
        if self.keyPressed or self.nfix > 49:
            return True
        return False

    def __str__(self):
       return objsASCII(self.objects)