__author__ = 'Ryan M. Hope <rmh3093@gmail.com>'

from pybrain.rl.agents import LearningAgent
from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer
from scipy import array
from helpers import *
import random
import math


class Human( object ):

    def __init__( self, type = 1 ): # learner = None, 
        self.type = type
        self.saccade_types = 3
        if self.type == 1:
            self.saccade_types = 6 #Highest, nearest, farthest saccades for sal and unc maps
	elif self.type == 0: #random fixation locations
	    self.saccade_types = 9
        self.button_types = 2 #self.learner.num_actions - self.saccade_types
	self.gauss_blur = 0.7
        self.color_x2coeff = 0.035 #default: 0.035
        self.shape_x2coeff = 0.035 #default: 0.3
        self.coeffofvar = 0.7 #default: 0.7
	self.colorConf = 0.0 #default: 0.0
	self.shapeConf = 0.0 #default: 0.0

    def newEpisode( self ):
        self.objects = self.uncertainty = self.saliency = None
        self.rx = np.linspace( -7, 7, 8 )
        self.ry = np.linspace( -5, 5, 6 )
        self.x, self.y = np.meshgrid( self.rx, self.ry )
        self.fixation_location = ( 0, 0 )
        self.targetVisible = 0
        self.nfix = 1
        #self.actionsTaken = []
        self.scanpath = []
        self.colorConf = 0.0 #default: 0.0
	self.shapeConf = 0.0 #default: 0.0

    def integrateObservation( self, obs ):
        """
        This function updates the saliency and uncertainty maps after a 
        new fixation then computes the peaks of each map. The peaks along with
        other state information get passed on to the learning agent.
        """

        if self.objects == None:
            self.objects = obs

        self.objects = score( apply_availability( self.objects, self.fixation_location, self.color_x2coeff, self.shape_x2coeff, self.coeffofvar ), 'RO' )
        self.activation = pmap( self.x, self.y, self.objects, score = 8, s = self.gauss_blur )
        self.uncertainty = pmap( self.x, self.y, self.objects, score = 9, s = self.gauss_blur )
        self.saliency = pmap( self.x, self.y, self.objects, score = 10, s = self.gauss_blur )

        # If all features of target are visible let the agent know so that
        # it can learn to end a trial
        if targetVisible( self.objects ):
            self.targetVisible = 1

        #what are we doing here? Are we providing the state to a learning agent?:
        if self.type == 1 or self.type == 0:
            maps = np.append( detect_peaks( self.uncertainty ).flatten(), detect_peaks( self.saliency ).flatten() )
        elif self.type == 2:
            maps = detect_peaks( self.saliency ).flatten()
        elif self.type == 3:
            maps = detect_peaks( self.uncertainty ).flatten()
        elif self.type == 4:
            maps = detect_peaks( self.activation ).flatten()
        return np.append( [self.nfix, self.fixation_location[0], self.fixation_location[1], self.targetVisible], maps )

    def getAction( self ):
        #action = int(self.lastaction)# this must change to a probability of actions...
        #print '  type = %i' % self.type
	if self.type > 0:
            action = selectAction()
        else:
            action = 8
        #print '  Action = %s' % action  
	self.scanpath.append( action )
	#print '    %s' % self.actionsTaken

        if self.type == 1:
            max1 = get_maxima( self.uncertainty )
            max2 = get_maxima( self.saliency )
        elif self.type == 2:
            max1 = get_maxima( self.saliency )
        elif self.type == 3:
            max1 = get_maxima( self.uncertainty )
        elif self.type == 4:
            max1 = get_maxima( self.activlearneration )

        newFix = False
        if ( action == 0 ):
            newFix = get_highest( max1, self.fixation_location )
        elif ( action == 8 ):
            newFix = get_rand_loc()
            #print '  action = %i' % action
            #print '  fix = %i,%i' % (newFix[1],newFix[0])
        elif ( action == 1 ):
            newFix = get_nearest( max1, self.fixation_location )
        elif ( action == 2 ):
            newFix = get_farthest( max1, self.fixation_location )
        if self.type == 1:
            if ( action == 3 ):
                newFix = get_highest( max2, self.fixation_location )
                #print '  fix = %i,%i' % (newFix[1],newFix[0]) 
            elif ( action == 4 ):
                newFix = get_nearest( max2, self.fixation_location )
                #print '  fix = %i,%i' % (newFix[1],newFix[0])
            elif ( action == 5 ):
                newFix = get_farthest( max2, self.fixation_location )
                #print '  fix = %i,%i' % (newFix[1],newFix[0])

        if action < self.saccade_types:
            # Convert grid location of maxima to x,y in visual angle
	    #print 'action = %i, nFix = %i' % (action, self.nfix)
	    #print '  newFix x? = %i, y? = %i' % (newFix[1],newFix[0])
            self.fixation_location = ( self.rx[newFix[1]], self.ry[newFix[0]] ) #CWM 8 Feb 2012 changed self.rx[newFix[0]] to self.ry[newFix[0]]
	    #print '    action %i  newLoc=[%i,%i]' % (action, self.fixation_location[0],self.fixation_location[1])
            # Update the fixation count
            #self.nfix += 1    
            return ( -1, 7 - self.saccade_types, self.targetVisible ) #changed self.learner.num_actions to 7
        #else:
        #    return (action-self.saccade_types,7-self.saccade_types, self.targetVisible) #changed self.learner.num_actions to 7

    def TASR_threshExceeded( self, threshold ):
	totColor = get_all_colors( self.objects )
	totShape = get_all_shapes( self.objects )
        allEncColors = get_all_enc_colors( self.objects )
        allEncShapes = get_all_enc_shapes( self.objects )
	encSameColor = get_same_enc_colors( self.objects )
	encSameShape = get_same_enc_shapes( self.objects )
        compObs = get_comp_items( self.objects )


        # ratio approximation with diminishing return thresholded TASR
        if self.colorConf == 0.0 and self.shapeConf == 0.0:
                self.colorConf = ( encSameColor / allEncColors ) * ( compObs / 48 )
                self.shapeConf = ( encSameShape / allEncShapes ) * ( compObs / 48 )
                #print '       cConf = %f   sConf = %f' % (self.colorConf,self.shapeConf)
                return False

        if allEncColors > 0.0:
		cConf = ( encSameColor / allEncColors ) * ( compObs / 48 )
                dimRet = math.fabs( self.colorConf - cConf )
                #print '       cConf = %f   oldcConf = %f   dimRet = %f' % (cConf,self.colorConf,dimRet)
                self.colorConf = cConf
                if dimRet <= threshold:
			return True

        if allEncShapes > 0.0:
		sConf = ( encSameShape / allEncShapes ) * ( compObs / 48 )
                dimRet = math.fabs( self.shapeConf - sConf )
                #print '     sConf = %f   oldsConf = %f   dimRet = %f' % (sConf,self.shapeConf,dimRet)
                self.shapeConf = sConf
                if dimRet <= threshold:
                        return True
        #=================================================================

        #Race with perfect and/or noisy knowledge of all like shapes & colors
        '''
	if totShape > 0.0:
                #No noise
		self.shapeConf = encShape / totShape
		#Noisy:
                #self.shapeConf = (encShape / totShape) + random.normalvariate(0.0, 0.1)
		#print '    shape conf = %f' % self.shapeConf
		if self.shapeConf >= threshold:
	   		return True        
	if totColor > 0.0:
                #No noise
		self.colorConf = encColor / totColor
		#Noisy:
                #self.colorConf = (encColor / totColor) + random.normalvariate(0.0, 0.1)
		#print '    color conf = %f' % self.colorConf
		if self.colorConf >= threshold:
			return True
        '''
        #=================================================================
        return False




class HumanAgent( LearningAgent, Human ):

    def __init__( self, module, learner = None, type = 1 ):
        super( HumanAgent, self ).__init__( module, learner )
        Human.__init__( self, learner, type )
        if learner:
            self.learner.explorer.agent = self

    def newEpisode( self ):
        super( HumanAgent, self ).newEpisode()
        Human.newEpisode( self )

    def integrateObservation( self, obs ):
        super( HumanAgent, self ).integrateObservation( Human.integrateObservation( self, obs ) )

    def getAction( self ):
        super( HumanAgent, self ).getAction()
        return Human.getAction( self )

class HumanAgent_LinearFA( LinearFA_Agent, Human ):

    laststate = None

    def __init__( self, learner, type = 1, **kwargs ):
        super( HumanAgent_LinearFA, self ).__init__( learner, **kwargs )
        Human.__init__( self, learner, type )

    def newEpisode( self ):
        super( HumanAgent_LinearFA, self ).newEpisode()
        Human.newEpisode( self )

    def integrateObservation( self, obs ):
        super( HumanAgent_LinearFA, self ).integrateObservation( Human.integrateObservation( self, obs ) )

    def getAction( self ):
        super( HumanAgent_LinearFA, self ).getAction()
        return Human.getAction( self )
