__author__ = 'Ryan M. Hope <rmh3093@gmail.com>'

from pybrain.rl.agents import LearningAgent
from helpers import *

class HumanAgent(LearningAgent):
    
    def __init__(self, module, learner = None):
        LearningAgent.__init__(self, module, learner)
        self.reset()
        
    def reset(self):
        LearningAgent.reset(self)
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
        self.uncertainty = pmap(self.x,self.y,self.objects,score=9)
        self.saliency = pmap(self.x,self.y,self.objects,score=10)
        
        # If all features of target are visible let the agent know so that
        # it can learn to end a trial
        if targetVisible(self.objects):
            self.targetVisible = 1
        
        maps = np.append(detect_peaks(self.uncertainty).flatten(),
                         detect_peaks(self.saliency).flatten())
        maps = np.append([self.nfix,self.targetVisible,self.fixation_location[0],self.fixation_location[1]],maps)
        
        LearningAgent.integrateObservation(self, maps)

    def getAction(self):
        LearningAgent.getAction(self)
        action = int(self.lastaction)
        maxu = get_maxima(self.uncertainty)
        maxs = get_maxima(self.saliency)
        newFix = False
        if (action==0): # Set new fix to highest uncertainty maxima
            newFix = get_highest(maxu)
        elif (action==1): # Set new fix to nearest uncertainty maxima
            newFix = get_nearest(maxu, self.fixation_location)
        elif (action==2): # Set new fix to farthest uncertainty maxima
            newFix = get_farthest(maxu, self.fixation_location)
        elif (action==3): # Set new fix to highest saliency maxima
            newFix = get_highest(maxs)
        elif (action==4): # Set new fix to nearest saliency maxima
            newFix = get_nearest(maxs, self.fixation_location)
        elif (action==5): # Set new fix to farthest saliency maxima
            newFix = get_farthest(maxs, self.fixation_location)

        if action<6:
            # Convert grid location of maxima to x,y in visual angle
            self.fixation_location = (self.rx[newFix[1]],self.rx[newFix[0]])

            # Update the fixation count
            self.nfix += 1
            
        return int(self.lastaction)