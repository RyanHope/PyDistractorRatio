#!/usr/bin/env python

__author__ = 'Ryan M. Hope <rmh3093@gmail.com>'

import os, sys, atexit
from environment import DistractorRatio
from human import HumanAgent, HumanAgent_LinearFA, HumanExplorer
from experiment import CustomEpisodicExperiment
from pybrain.rl.learners.valuebased.linearfa import QLambda_LinFA
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

if len(sys.argv) < 2 or (int(sys.argv[1])<0 or int(sys.argv[1])>4):
    print 'Must supply a model type:\n\t1 = Uncert&Salience, 2 = Salience, 3 = Uncert, 4 = Activation!'
    sys.exit()

if len(sys.argv) < 3:
    print 'Must supply an output file!'
    sys.exit()

type = int(sys.argv[1]) # 1 = Uncert&Salience, 2 = Salience, 3 = Uncert, 4 = Activation
env = DistractorRatio() # Create an instance of the D-R task
# Create an action/value neural net with an state space of 100 and an action space of 8
if type == 1:
    learner = QLambda_LinFA(8,100)
else:
    learner = QLambda_LinFA(5,52)

learner.batchMode = False    
agent = HumanAgent_LinearFA(learner, type) # Create an agent that learns with QLambda_LinFA
experiment = CustomEpisodicExperiment(env, agent) # Put the agent in the environment

if len(sys.argv) == 4:
    print 'Loading saved net...'
    module.network = NetworkReader.readFrom(sys.argv[3])

def save(history):
    """
    This function gets called after each training/testing block or when the
    script gets closed. It saves the neural net and RL history of the agent so
    that it can be restored or reused in another model.
    """
    base = os.path.splitext(sys.argv[2])[0]
    fileObject = open(base+'.history', 'w')
    pickle.dump(history, fileObject)
    fileObject.close()

# This registers a function that will get called when the script closes.
atexit.register(save, agent.history)

performance = 0

# Touch the model output file to make sure its empty
f = open(sys.argv[2],'w')
f.close()

while(True):
    print '=============== TRAINING ===================='
    performance = performance + 1
    # Train the model using distractor ratios 1,7,9,15 for both target present
    # and target absent trials, then update the neural net. Do this 5 times in a row.
    for i in range(0,1):
        # Set the task in training mode so that is knows to use the reduced set
        # of distractor ratios
        experiment.task.train = False
        # Generate a new randomized pool of distractor ratios
        experiment.task.newPool()
        # Do 1 block of 8 distractors ratios, this will use up all the distractor
        # ratios in the pool.
        experiment.doEpisodes(30)
        # Update the neural net based on all past experience.
        print
    agent.learn()
    
    print '=============== TESTING ===================='
    # Disable training mode. This will tell the environment to use the complete
    # set of 15 distractor ratios when generating new pools.
    experiment.task.train = False
    # Test the model using all 15 distractor ratios for both target present and
    # target absent trials. Do this 20 times without updating the neural net.
    for i in range(0,1):
        # Generate a new randomized pool of distractor ratios
        experiment.task.newPool()
        # Switch to the testing agent so that testing trials don't get added to
        # the agents learning history
        experiment.agent.learning = False
        # Do 30 trials (which is 1 of each distractor ratio for both target
        # absent and target present trials) and collect some stats.
        x,y,z,w = experiment.doEpisodes(30)
        # Write the stats to a file.
        f = open(sys.argv[2],'a')
        for i in range(0,len(x)):
            f.write('%d\t%d\t%f\t%d\t%f\n' % (performance,y[i],sum(x[i]),z[i],mean(w[i])))
        f.close()
    # Save the trained neural network and the agent RL history to files.
    save(agent.history)
    
    env.delay = False    
    # Set the agent back to the normal learning agent
    experiment.agent.learning = False
