#!/usr/bin/env python

__author__ = 'Chris Myers, myers.christopher@gmail.com'
# to run: app      file       type    outputfile              blocks
# to run: python DR-MC_Model.py 1 /home/chris/Desktop/test.xls 2
# Model Parameters:
#   sal_crit = 0.95
#   unc_crit = 0.95
#   stop_crit = 0.60

import os, sys, atexit
from environment import DistractorRatio
from human import HumanAgent, HumanAgent_LinearFA, Human #MCAgent
from experiment import CustomEpisodicExperiment
from helpers import get_peak_count, stop_absent, sum_avail_features
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

type = int( sys.argv[1] ) # 1 = Uncert&Salience, 2 = Salience, 3 = Uncert, 4 = Activation
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
f.write( 'block,trial,ratio,target,response,nFix,reward,sal_peaks,\n' )
f.close()

sal_crit = 0.95
unc_crit = 0.95
stop_crit = 0.60
max_fixes = 48
block = 0
scanpath = []

for i in range( 0, BlockTrials ):
    block = block + 1
    trial = 0
    print 'Block %d' % ( block )
    experiment.task.newPool() # Generate a new randomized pool of 30 distractor ratios
    nTrials = len( experiment.task.pool )
    for dummy in range( 0, nTrials ):
	trial = trial + 1
	#print ' Trial %i' % trial
        trialFixes = []
	sal_peaks = 0
	experiment.agent.newEpisode()
        experiment.task.reset()  #prepares environment w/ objects

	# Open File.
	f = open( sys.argv[2], 'a' )
	f.write( '%d,%d,%d,%s,' % ( block, trial, experiment.task.ratio, experiment.task.hasTarget ) )
	while not experiment.task.isFinished():
		experiment.stepid += 1
		observ = experiment.task.getObservation()
        	experiment.agent.integrateObservation( observ )
                experiment.agent.scanpath.append( experiment.agent.fixation_location )
		if sal_peaks == 0:
			sal_peaks = get_peak_count( experiment.agent.saliency )
			#unc_peaks = get_peak_count(experiment.agent.uncertainty, unc_crit)
                	#print '   peaks = %f' % sal_peaks

		if stop_absent( experiment.agent.objects, stop_crit ):
		#previous if conditional: get_peak_count(experiment.agent.saliency, unc_crit) <= 5: #experiment.task.nfix >= absentCrit:
			experiment.task.performAction( ( 'absent', 1, experiment.agent.targetVisible ) )
			response = 'absent'
			reward = experiment.task.getReward()
			experiment.task.done = True
		elif experiment.agent.targetVisible == 1:
			experiment.task.performAction( ( 'present', 1, 1 ) )
			response = 'present'
			reward = experiment.task.getReward()
			experiment.task.done = True
		# A Catch for target absent trials:
		elif experiment.task.nfix > max_fixes:
			experiment.task.performAction( ( 'absent', 1, experiment.agent.targetVisible ) )
			response = 'failure'
			#reward = experiment.task.getReward()
			experiment.task.done = True

		action = experiment.agent.getAction()
        	experiment.task.performAction( action )

		experiment.agent.scanpath.append( experiment.agent.fixation_location )

	#'block,trial,ratio,target,response,nFix,reward\n'
        f.write( '%s,%d,%f,%i,' % ( response, experiment.task.nfix, reward, sal_peaks ) )
	nFixes = len( trialFixes )
        scan = len( experiment.agent.scanpath )
        experiment.agent.scanpath.reverse()
        for dummy in range( 0, scan ):
		item = experiment.agent.scanpath.pop()
		if isinstance( item, tuple ):
			f.write( '[%s--%s]-->,' % ( item[0], item[1] ) )
		elif isinstance( item, int ):
			f.write( '%s-->,' % item )
	f.write( '\n' )
        f.close()
        #print
