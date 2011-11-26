__author__ = 'Ryan M. Hope <rmh3093@gmail.com>'

import sys
from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment
from helpers import *
from math import ceil

class DistractorRatio(Environment, Named):
    
    objects = None
    fixation_location = (0,0)
    keypress = None
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
        self.keypress = None
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
        print ' %d' % (action),
        
        if (action==6): # Press A key
            self.keypress = 'A'
        elif (action==7): # Press P key
            self.keypress = 'P'
        else:
            self.nfix += 1

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
        if self.keypress != None:
            if self.keypress != None:
                self.done = True # Keypress marks end of trial
            if self.keypress == 'P' and self.hasTarget:
                # Give a positive reward that decreases with number of fixations
                return 100.0/self.nfix
            elif self.keypress == 'A' and not self.hasTarget:
                # Give a positive reward that decreases with number of fixations
                return 100.0/self.nfix
            else:
                # Give a negative reward equal to the number of fixations
                return -1*self.nfix
        return 0.0
        
    def isFinished(self):
        """
        This returns True when the experiment/task episode is over. Currently
        an episode is over when the number of fixations exceeds 200 or if a key
        was pressed by the agent at which point the variable done is set to true.
        The done variable is set to true in the getReward() function and not the
        performAction() function to ensure the reward/penalty is received by the
        the agent.
        """
        if self.done or self.nfix > 200:
            return True
        return False