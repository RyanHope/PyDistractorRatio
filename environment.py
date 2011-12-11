__author__ = 'Ryan M. Hope <rmh3093@gmail.com>'

import sys
from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment
from helpers import *
from math import ceil

class DistractorRatio(Environment, Named):
    
    objects = None
    fixation_location = (0,0)
    keypress = -1
    nfix = 1
    ratio = 0
    done = False
    saliency = None
    uncertainty = None
    train = False
    hasTarget = False

    def __init__(self, **args):
        self.setArgs(**args)
        self.newPool()

    def newPool(self):
        if self.train:
            self.pool = sample([1,7,9,15,16,22,24,30],8)
        else:
            self.pool = sample(range(1,31),30)

    def reset(self):
        self.done = False
        self.nfix = 1
        self.keypress = -1
        if len(self.pool) < 1:
            self.newPool()
        p = self.pool.pop()
        if (ceil(p/15.0)>1):
            self.hasTarget = True
            self.ratio = p-15
        else:
            self.hasTarget = False
            self.ratio = p
        
        print '~~~~~~~~~~~~~~~ Ratio %d, Has Target %d ~~~~~~~~~~~~~~~~~~~~~' % (self.ratio,self.hasTarget)
        while True:
            self.objects = generateObjects(self.ratio, self.hasTarget)
            if not targetVisible(apply_availability(self.objects, (0,0))):
                return
            
    def performAction(self, action):
        """
        This function gets called after an action is selected. In this function
        is where the agent affects the world/state by performing some action.
        In this environment there are 8 actions. Three actions correspond to
        moving the eye to the highest, nearest and farthest saliency maxima.
        Three actions correspond to moving the eye to the highest, nearest and 
        farthest uncertainty maxima. The final two actions correspond to button
        presses, either A for target absent or P for target present.
        """
        
        if action==(None,None):
            self.nfix += 1
        self.keypress = action

    def getObservation(self):
        """
        This is called before every action. Since the environment is static,
        this always returns the same thing... the vector of objects in the
        search display.
        """
        return self.objects

    def getReward(self):
        """
        This gets called after every action. In this task we always return 0
        reward unless the task is over at which point the reward/penalty is 
        based on number of fixations that were made before the task was over. 
        """
        
        """
        if self.keypress != -1:
            self.done = True
            if self.keypress == 1:
                if self.hasTarget:
                    return 50
                else:
                    return -25
            elif self.keypress == 0:
                if not self.hasTarget:
                    return 50
                else:
                    return -25
        return -1
        """
        
        kp,flag = self.keypress
        
        if self.nfix > 100:
            self.done = True
            return 0
        
        if kp == None:
            return 0
        
        self.done = True
        
        if self.hasTarget:
            if flag:
                if kp:
                    reward = 150.0/self.nfix
                else:
                    reward = 50.0/self.nfix
            else:
                if kp:
                    reward = -3.0*self.nfix
                else:
                    reward = -6.0*self.nfix
        else:
            if kp:
                reward = -4*self.nfix
            else:
                reward = 10.0/self.nfix
        
        return reward
        
        
    def isFinished(self):
        """
        This returns True when the experiment/task episode is over. Currently
        an episode is over when the number of fixations exceeds 200 or if a key
        was pressed by the agent at which point the variable done is set to true.
        The done variable is set to true in the getReward() function and not the
        performAction() function to ensure the reward/penalty is received by the
        the agent.
        """
        if self.done:
            return True
        return False