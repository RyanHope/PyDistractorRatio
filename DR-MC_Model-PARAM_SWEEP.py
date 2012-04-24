#!/usr/bin/env python

__author__ = 'Chris Myers, myers.christopher@gmail.com'
# to run: app      file       type    outputfile              blocks   
# to run: python DR-MC_Model-PARAM_SWEEP.py 1 ~/Desktop/DimRetTASR-Data.csv 1000 

import os, sys, atexit
from environment import DistractorRatio
from human import HumanAgent, HumanAgent_LinearFA, Human #MCAgent
from experiment import CustomEpisodicExperiment
from helpers import get_peak_count, stop_absent, sum_avail_features, get_fixated_obj_color, get_fixated_obj_shape
#from pybrain.rl.learners.valuebased.linearfa import QLambda_LinFA

try:
    from pybrain.tools.customxml.networkwriter import NetworkWriter
    from pybrain.tools.customxml.networkreader import NetworkReader
except:
    from pybrain.tools.xml.networkwriter import NetworkWriter
    from pybrain.tools.xml.networkreader import NetworkReader

try:
   import cPickle as pickle
except:
   import pickle

from numpy import array, arange, meshgrid, pi, zeros, mean

if len( sys.argv ) < 2 or ( int( sys.argv[1] ) < 0 or int( sys.argv[1] ) > 4 ):
    print 'Must supply a model type:\n\t1 = Uncert&Salience, 2 = Salience, 3 = Uncert, 4 = Activation!'
    sys.exit()

if len( sys.argv ) < 3:
    print 'Must supply an output file!'
    sys.exit()

if len( sys.argv ) < 4:
    print 'Must supply the number of trials for the model!'
    sys.exit()

type = int( sys.argv[1] ) # 0 = random location, 1 = Uncert&Salience, 2 = Salience, 3 = Uncert, 4 = Activation
BlockTrials = int( sys.argv[3] ) # number of trial blocks of 30 the model will complete
#BlockTrials = 2
env = DistractorRatio() # Create an instance of the D-R task

agent = Human( type ) # Create an agent that learns with QLambda_LinFA
experiment = CustomEpisodicExperiment( env, agent ) # Put the agent in the environment

# Touch the model output file to make sure its empty
f = open( sys.argv[2], 'w' )
f.close()

# 1. Generate a trial
# 2. Produce perceptual map (pmap) from trial && model fixation location
# 3. Test stopping rule for 'present' trials
#     IF both target features are available; THEN respond, reward trial && GOTO 1; ELSE continue
# 4. Test stopping rule for 'absent' trials
#     IF 'nFixes >= Criterion'; THEN respond, reward trial, && GOTO 1; ELSE continue
# 5. Select action based on parameterized saccade-selection probabilities
# 6. GOTO 2.
f = open( sys.argv[2], 'a' )
f.write( 'model,STOPthresh,noise,blur,color,shape,block,trial,ratio,target,response,EnFix,reward,accuracy,sal_peaks,LikeColor,LikeShape,IG\n' )
f.close()

# Model Parameters:
mod_title = "40-15-15-10-10-10"
max_fixes = 48
TASRthresholdParams = [0.01]
scanpath = []
coeffofvarparams = [0.7] #default = 0.7
colorcoeffparams = [0.035] #default = 0.035
shapecoeffparams = [0.035] #default = 0.3
#sParams = [0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725]
#prop_uncovered = [0.15, 0.2, 0.25, 0.3] #proportion of features that are not uncovered
experiment.agent.gauss_blur = 0.7

for p in range( 0, len( TASRthresholdParams ) ):
    TASRthreshold = TASRthresholdParams.pop()
    #experiment.agent.coeffofvar = coeffofvarparams.pop()
    #prop_for_stop = prop_uncovered.pop()
    print 'Type = %i' % experiment.agent.type
    print ' Thresh %f' % TASRthreshold
    #print 'Stopping Proportion: %f' % prop_for_stop
    block = 0
    for i in range( 0, BlockTrials ):
	block = block + 1
	trial = 0
	print '  Block %d' % ( block )
	experiment.task.newPool() # Generate a new randomized pool of 30 distractor ratios
	nTrials = len( experiment.task.pool )
	for dummy in range( 0, nTrials ):
	    trial = trial + 1
	    nLikeColorFixes = 0
	    nLikeShapeFixes = 0
	    print '   Trial %i' % trial
	    trialFixes = []
	    sal_peaks = 0
	    unc_peaks = []
	    avail_feats = 0
	    sumd_feats = []
	    item_feats_avail = []
	    experiment.agent.newEpisode()
	    experiment.task.reset()  #prepares environment w/ objects
            accuracy = None
	    #print 'Distractor Ratio: %i' % experiment.task.ratio

	    # Open File.
	    f = open( sys.argv[2], 'a' )
	    f.write( '%s,%f,%f,%f,%f,%f,%d,%d,%d,%s,' % ( mod_title, TASRthreshold, experiment.agent.coeffofvar, experiment.agent.gauss_blur, experiment.agent.color_x2coeff, experiment.agent.shape_x2coeff, block, trial, experiment.task.ratio, experiment.task.hasTarget ) )
	    while not experiment.task.isFinished():
		    experiment.stepid += 1
		    observ = experiment.task.getObservation()
		    experiment.agent.integrateObservation( observ ) #increases the fixation count
   		    hidden_feats = sum_avail_features( experiment.agent.objects )

	            #== debugging & data ================================
	            experiment.agent.scanpath.append( experiment.agent.fixation_location )
                    nLikeColorFixes = get_fixated_obj_color( experiment.agent.objects, nLikeColorFixes, experiment.agent.fixation_location )
                    nLikeShapeFixes = get_fixated_obj_shape( experiment.agent.objects, nLikeShapeFixes, experiment.agent.fixation_location )
	            #===================================================

		    if experiment.agent.TASR_threshExceeded( TASRthreshold ): #experiment.task.nfix >= max_fixes:
			    experiment.task.performAction( ( 'absent', 1, experiment.agent.targetVisible ) )
			    response = 'absent'
			    reward = experiment.task.getReward()
                            if reward > 0:
                               accuracy = "correct"
                            else:
                               accuracy = "error"
			    experiment.task.done = True
		    elif experiment.agent.targetVisible == 1:
			    experiment.task.performAction( ( 'present', 1, 1 ) )
			    response = 'present'
			    reward = experiment.task.getReward()
                            if reward > 0:
                               accuracy = "correct"
                            else:
                               accuracy = "error"
			    experiment.task.done = True
		    # A Catch for target absent trials:

		    elif experiment.task.nfix >= max_fixes:
			    experiment.task.performAction( ( 'failure', 1, experiment.agent.targetVisible ) )
			    response = 'failure'
                            accuracy = "failure"
			    reward = experiment.task.getReward()
			    experiment.task.done = True

		    if hidden_feats > 1 and experiment.task.done == False:
			    action = experiment.agent.getAction()
		            experiment.task.performAction( action )

	    #== debugging & data ================================	

	    #'block,trial,ratio,target,response,nFix,reward\n'
	    f.write( '%s,%d,%s,%s,%i,%s,%s,' % ( response, experiment.task.nfix, reward, accuracy, sal_peaks, nLikeColorFixes, nLikeShapeFixes ) )
	    nFixes = len( trialFixes )
	    #gains = len(sumd_feats)
	    #sumd_feats.reverse()

	    #===================================================
	    f.write( '\n' )
	    f.close()
