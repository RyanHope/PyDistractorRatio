#!/usr/bin/env python

import arac
from environment import DistractorRatio
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork

from numpy import array, arange, meshgrid, pi, zeros, mean
from matplotlib import pyplot as plt

plt.ion()

env = DistractorRatio()

module = ActionValueNetwork(65, 65)
#module.convertToFastNetwork()
learner = NFQ()
#learner.offPolicy = True
learner.explorer.epsilon = 0.4
agent = LearningAgent(module, learner)
testagent = LearningAgent(module, None)
experiment = EpisodicExperiment(env, agent)

def plotPerformance(values, fig):
    plt.figure(fig.number)
    plt.clf()
    plt.plot(values, 'o-')
    plt.gcf().canvas.draw()

performance = []

pf_fig = plt.figure()

while(True):
    # one learning step after one episode of world-interaction
    print
    experiment.task.random = True
    #experiment.task.ratio = 0
    experiment.doEpisodes(1)
    agent.learn(1)
    experiment.task.random = True
    print
    
    # test performance (these real-world experiences are not used for training)
    experiment.agent = testagent
    r = mean([sum(x) for x in experiment.doEpisodes(5)])
    env.delay = False
    testagent.reset()
    experiment.agent = agent

    performance.append(r)
    plotPerformance(performance, pf_fig)

    print "reward avg", r
    print "explorer epsilon", learner.explorer.epsilon
    print "num episodes", agent.history.getNumSequences()
    print "update step", len(performance)