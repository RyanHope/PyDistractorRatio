#!/usr/bin/env python

from environment import DistractorRatio
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import Experiment
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners import QLambda
from pybrain.rl.explorers import BoltzmannExplorer

from numpy import array, arange, meshgrid, pi, zeros, mean
from matplotlib import pyplot as plt

plt.gray()
plt.ion()

env = DistractorRatio()

table = ActionValueTable(84, 83)
table.initialize(1.)
learner = QLambda()
agent = LearningAgent(table, learner)
experiment = Experiment(env, agent)

for i in range(1000):

    # interact with the environment (here in batch mode)
    experiment.doInteractions(100)
    agent.learn()
    agent.reset()

    # and draw the table
    pylab.pcolor(table.params.reshape(81,4).max(1).reshape(9,9))
    pylab.draw()