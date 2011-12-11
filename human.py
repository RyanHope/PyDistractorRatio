__author__ = 'Ryan M. Hope <rmh3093@gmail.com>'

from pybrain.rl.agents import LearningAgent
from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer
from scipy import array
from helpers import *

class HumanExplorer(EpsilonGreedyExplorer):
    
    agent = None

    def _forwardImplementation(self, inbuf, outbuf):
        EpsilonGreedyExplorer._forwardImplementation(self, inbuf, outbuf)
        # If target is visible then stop exploring and end trial
        if self.agent.targetVisible:
            outbuf[:] = array([self.module.numActions-1])
      
class Human(object):
    
    def __init__(self, learner = None, type=1):
        self.type = type
        if self.type == 1:
            self.nactions = 8
        else:
            self.nactions = 5
            
    def newEpisode(self):
        self.objects = self.uncertainty = self.uncertainty = None
        self.rx = np.linspace(-7,7,8)
        self.ry = np.linspace(-5,5,6)
        self.x,self.y = np.meshgrid(self.rx,self.ry)
        self.fixation_location = (0,0)
        self.targetVisible = 0
        self.nfix = 1
        
    def integrateObservation(self, obs):
        """
        This function updates the saliency and uncertainty maps after a 
        new fixation then computes the peaks of each map. The peaks along with
        other state information get passes on to the learning agent.
        """
        
        if self.objects == None:
            self.objects = obs
            
        self.objects = score(apply_availability(self.objects, self.fixation_location),'RO')
        self.activation = pmap(self.x,self.y,self.objects,score=8)
        self.uncertainty = pmap(self.x,self.y,self.objects,score=9)
        self.saliency = pmap(self.x,self.y,self.objects,score=10)
        
        # If all features of target are visible let the agent know so that
        # it can learn to end a trial
        if targetVisible(self.objects):
            self.targetVisible = 1
        
        if self.type == 1:
            maps = np.append(detect_peaks(self.uncertainty).flatten(),detect_peaks(self.saliency).flatten())
        elif self.type == 2:
            maps = detect_peaks(self.saliency).flatten()
        elif self.type == 3:
            maps = detect_peaks(self.uncertainty).flatten()
        elif self.type == 4:
            maps = detect_peaks(self.activation).flatten()
        return np.append([self.nfix,self.fixation_location[0],self.fixation_location[1],self.targetVisible],maps)
    
    def getAction(self):
        action = int(self.lastaction)
        
        if self.type == 1:
            max1 = get_maxima(self.uncertainty)
            max2 = get_maxima(self.saliency)
        elif self.type == 2:
            max1 = get_maxima(self.saliency)
        elif self.type == 3:
            max1 = get_maxima(self.uncertainty)
        elif self.type == 4:
            max1 = get_maxima(self.activation)
            
        newFix = False
        if (action==0):
            newFix = get_highest(max1)
        elif (action==1):
            newFix = get_nearest(max1, self.fixation_location)
        elif (action==2):
            newFix = get_farthest(max1, self.fixation_location)
        if self.type == 1:
            if (action==3):
                newFix = get_highest(max2)
            elif (action==4):
                newFix = get_nearest(max2, self.fixation_location)
            elif (action==5):
                newFix = get_farthest(max2, self.fixation_location)

        print ' %d' % (action),

        if action == self.nactions-1 or action == self.nactions-2:
            return (self.nactions - action - 1, self.targetVisible)
        else:
            # Convert grid location of maxima to x,y in visual angle
            self.fixation_location = (self.rx[newFix[1]],self.rx[newFix[0]])
            # Update the fixation count
            self.nfix += 1    
            return (None,None)
      
class HumanAgent(LearningAgent, Human):
    
    def __init__(self, module, learner = None, type=1):
        super(HumanAgent, self).__init__(module, learner)
        Human.__init__(self, learner, type)
        if learner:
            self.learner.explorer.agent = self
        
    def newEpisode(self):
        super(HumanAgent, self).newEpisode()
        Human.newEpisode(self)
        
    def integrateObservation(self, obs):        
        super(HumanAgent, self).integrateObservation(Human.integrateObservation(self, obs))

    def getAction(self):
        super(HumanAgent, self).getAction()
        return Human.getAction(self)
    
class HumanAgent_LinearFA(LinearFA_Agent, Human):
    
    def __init__(self, learner, type=1, **kwargs):
        super(HumanAgent_LinearFA, self).__init__(learner, **kwargs)
        Human.__init__(self, learner, type)
        
    def newEpisode(self):
        super(HumanAgent_LinearFA, self).newEpisode()
        Human.newEpisode(self)
        
    def integrateObservation(self, obs):        
        super(HumanAgent_LinearFA, self).integrateObservation(Human.integrateObservation(self, obs))

    def getAction(self):
        super(HumanAgent_LinearFA, self).getAction()
        return Human.getAction(self)