#!/usr/bin/env python

__author__ = 'Ryan M. Hope <rmh3093@gmail.com>'

import os, sys, atexit
from environment import DistractorRatio
from human import HumanAgent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork
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

if len(sys.argv) < 1:
    print 'Must supply an output file!'
    sys.exit()

class CustomEpisodicExperiment(EpisodicExperiment):
    """
    The extension of Episodic Experiment to handle blocked episodic tasks and
    custom logging. Normally, the episodic experiment just returns the collected
    rewards, but we want to collect information about the distractor ratio, target
    presence and the number of fixations.
    """

    def __init__(self, task, agent):
        # Init the parent class
        EpisodicExperiment.__init__(self, task, agent)
        
    def doEpisodes(self, number = 1):
        if self.doOptimization:
            self.optimizer.maxEvaluations += number
            self.optimizer.learn()
        else:
            all_rewards = []
            all_ratios = []
            all_tp = []
            all_fix = []
            for dummy in range(number):
                self.agent.newEpisode()
                rewards = []
                self.stepid = 0
                self.task.reset()
                print ' -->',
                while not self.task.isFinished():
                    r = self._oneInteraction()
                    rewards.append(r)
                all_rewards.append(rewards)
                all_ratios.append(self.task.ratio)
                all_tp.append(self.task.hasTarget)
                all_fix.append(self.task.nfix)
                print ' <--'

            return all_rewards,all_ratios,all_tp,all_fix

env = DistractorRatio() # Create an instance of the D-R task
# Create an action/value neural net with an state space of 100 and an action space of 8
module = ActionValueNetwork(100, 8)  
learner = NFQ()
learner.offPolicy = False # Disable off policy learning
learner.explorer.epsilon = 0.4
agent = HumanAgent(module, learner) # Create an agent that learns with NFQ
testagent = HumanAgent(module, None) # Create a testing agent
experiment = CustomEpisodicExperiment(env, agent) # Put the agent in the environment

if len(sys.argv) == 3:
    print 'Loading saved net...'
    module.network = NetworkReader.readFrom(sys.argv[2])

def save(history, net):
    """
    This function gets called after each training/testing block or when the
    script gets closed. It saves the neural net and RL history of the agent so
    that it can be restored or reused in another model.
    """
    base = os.path.splitext(sys.argv[1])[0]
    print 'Saving network to: ' + base + '.xml'
    NetworkWriter.writeToFile(net, base+'.xml')
    fileObject = open(base+'.history', 'w')
    pickle.dump(history, fileObject)
    fileObject.close()

# This registers a function that will get called when the script closes.
atexit.register(save, agent.history, module.network)

performance = 0

# Touch the model output file to make sure its empty
f = open(sys.argv[1],'w')
f.close()

while(True):
    print '=============== TRAINING ===================='
    performance = performance + 1
    # Train the model using distractor ratios 1,7,9,15 for both target present
    # and target absent trials, then update the neural net. Do this 5 times in a row.
    for i in range(0,5):
        # Set the task in training mode so that is knows to use the reduced set
        # of distractor ratios
        experiment.task.train = True
        # Generate a new randomized pool of distractor ratios
        experiment.task.newPool()
        # Do 1 block of 8 distractors ratios, this will use up all the distractor
        # ratios in the pool.
        experiment.doEpisodes(8)
        # Update the neural net based on all past experience.
        agent.learn(1)
        print
    
    print '=============== TESTING ===================='
    # Disable training mode. This will tell the environment to use the complete
    # set of 15 distractor ratios when generating new pools.
    experiment.task.train = False
    # Test the model using all 15 distractor ratios for both target present and
    # target absent trials. Do this 20 times without updating the neural net.
    for i in range(0,20):
        # Generate a new randomized pool of distractor ratios
        experiment.task.newPool()
        # Switch to the testing agent so that testing trials don't get added to
        # the agents learning history
        experiment.agent = testagent
        # Do 30 trials (which is 1 of each distractor ratio for both target
        # absent and target present trials) and collect some stats.
        x,y,z,w = experiment.doEpisodes(30)
        # Write the stats to a file.
        f = open(sys.argv[1],'a')
        for i in range(0,len(x)):
            f.write('%d\t%d\t%f\t%d\t%f\n' % (performance,y[i],sum(x[i]),z[i],mean(w[i])))
        f.close()
    # Save the trained neural network and the agent RL history to files.
    save(agent.history, module.network)
    
    env.delay = False    
    testagent.reset()
    # Set the agent back to the normal learning agent
    experiment.agent = agent
