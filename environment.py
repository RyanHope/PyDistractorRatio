__author__ = 'Ryan M. Hope <rmh3093@gmail.com>'

import sys
import random
from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment
from helpers import *
from math import ceil

class DistractorRatio( Environment, Named ):

    objects = None
    #fixation_location = (0,0)
    keypress = -1
    nfix = 1
    ratio = 0
    done = False
    saliency = None
    uncertainty = None
    train = False
    hasTarget = False
    targetVisible = False
    numActions = -1

    def __init__( self, **args ):
        self.setArgs( **args )
        self.newPool()

    def newPool( self ):
        if self.train:
            self.pool = sample( [1, 2, 7, 9, 14, 15, 16, 17, 22, 24, 29, 30], 12 )
        else:
            self.pool = sample( range( 1, 31 ), 30 )

    def reset( self ):
        self.done = False
        self.nfix = 1
        self.keypress = -1
        self.hasTarget = False
        self.targetVisible = False
        self.numActions = -1
        if len( self.pool ) < 1:
            self.newPool()
        p = self.pool.pop()
        if ( ceil( p / 15.0 ) > 1 ):
            self.hasTarget = True
            self.ratio = p - 15
        else:
            self.hasTarget = False
            self.ratio = p

        #print '~~~~~~~~~~~~~~~ Ratio %d, Has Target %d ~~~~~~~~~~~~~~~~~~~~~' % (self.ratio,self.hasTarget)
        while True:
            self.objects = generateObjects( self.ratio, self.hasTarget )
            return
            #if not targetVisible(apply_availability(self.objects, (0,0),)):
            #    return

    def performAction( self, action ):
        """
        This function gets called after an action is selected. In this function
        is where the agent affects the world/state by performing some action.
        In this environment there are 8 actions. Three actions correspond to
        moving the eye to the highest, nearest and farthest saliency maxima.
        Three actions correspond to moving the eye to the highest, nearest and 
        farthest uncertainty maxima. The final two actions correspond to button
        presses, either A for target absent or P for target present.
        """

        action, num_actions, target_visible = action
        #print '    action: %i, numacts: %i, targvis: %i' % (action, num_actions, target_visible)

        if action == -1:
            self.nfix += 1
	self.targetVisible = target_visible
	self.numActions = num_actions
        self.keypress = action

    def getObservation( self ):
        """
        This is called before every action. Since the environment is static,
        this always returns the same thing... the vector of objects in the
        search display.
        """
        return self.objects

#    def getReward(self):
#        if self.keypress != None:
#            if self.keypress != None:
#                self.done = True # Keypress marks end of trial
#            if self.keypress == 'P' and self.hasTarget:
#                return 100.0/self.nfix
#            elif self.keypress == 'A' and not self.hasTarget:
#                return 100.0/self.nfix
#            else:
#                return -1*self.nfix
#        return 0.0

    def getReward( self ):
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

        """
        accuracy = -1
        self.done = True
        if self.nfix>49:
            return 0
        elif self.keypress == -1:
            self.done = False
            return 0
        elif self.keypress == 1:
            accuracy = 1
        elif self.keypress == 0 and not self.hasTarget:
            accuracy = 1
        
        if self.nfix > 1:
            return 100*accuracy - 1*self.nfix
        else:
            return -10
        """
        '''
        if self.nfix>49:
            self.done = True
            return -0.001
        '''

        if self.keypress == -1:
            return 0

        self.done = True

	if self.hasTarget and self.keypress == 'present':
            accuracy = 100.0
            reward = accuracy / self.nfix
	if self.hasTarget and self.keypress == 'absent':
            accuracy = 0.0
            reward = accuracy / self.nfix
	if not self.hasTarget and self.keypress == 'present':
            accuracy = 0.0
            reward = accuracy / self.nfix
	if not self.hasTarget and self.keypress == 'absent':
            accuracy = 100.0
	    reward = accuracy / self.nfix
        if self.keypress == 'failure':
            reward = 'failure'
	return reward

        '''
        if self.hasTarget and self.targetVisible and self.keypress:
            return 100
        elif self.hasTarget and self.targetVisible and not self.keypress:
            return -10
        elif self.hasTarget and not self.targetVisible and self.keypress:
            return 1
        elif self.hasTarget and not self.targetVisible and not self.keypress:
            return -10
        elif not self.hasTarget and not self.targetVisible and self.keypress:
            return -100
        elif not self.hasTarget and not self.targetVisible and not self.keypress:
            return 10
        '''

    def isFinished( self ):
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
