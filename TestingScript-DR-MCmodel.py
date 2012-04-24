# Loading:
import os, sys, atexit
from environment import DistractorRatio
from human import HumanAgent, HumanAgent_LinearFA, Human
from experiment import CustomEpisodicExperiment
from helpers import get_peak_count, stop_absent
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import cPickle as pickle
from numpy import array, arange, meshgrid, pi, zeros, mean
type = 1
BlockTrials = 10
env = DistractorRatio()
agent = Human( type )
experiment = CustomEpisodicExperiment( env, agent )
block = 0
absentCrit = 14
sal_crit = 0.95
unc_crit = 0.95
stop_crit = 0.60
max_fixes = 48

#Model Behavior:
block = block + 1
trial = 0
experiment.task.newPool()
nTrials = len( experiment.task.pool )
trial = trial + 1
experiment.agent.newEpisode()
experiment.task.reset()
trialFixes = []
experiment.task.isFinished()
experiment.stepid += 1
observ = experiment.task.getObservation()
experiment.agent.integrateObservation( observ )
action = experiment.agent.getAction()
experiment.task.performAction( action )

#Testing Uncertainty Map's Dynamism:
experiment.task.ratio

experiment.agent.fixation_location
sum_avail_features( experiment.agent.objects )
sum_avail_features( experiment.task.objects )
get_peak_count( experiment.agent.uncertainty )
maxes = get_maxima( experiment.agent.uncertainty )
get_highest( maxes )
get_nearest( maxes, experiment.agent.fixation_location )
get_farthest( maxes, experiment.agent.fixation_location )

#Useful function calls:
experiment.agent.fixation_location
experiment.agent.objects
experiment.task.ratio
experiment.task.hasTarget
experiment.task.nfix
experiment.agent.targetVisible

#Saccade & Fixation Information:
experiment.task.nfix
experiment.agent.actionsTaken
scanpath
trialFixes
