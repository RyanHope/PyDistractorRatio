from pybrain.rl.experiments import EpisodicExperiment

class CustomEpisodicExperiment( EpisodicExperiment ):
    """
    The extension of Episodic Experiment to handle blocked episodic tasks and
    custom logging. Normally, the episodic experiment just returns the collected
    rewards, but we want to collect information about the distractor ratio, target
    presence and the number of fixations.
    """

    def doEpisodes( self, number = 1 ):
        if self.doOptimization:
            self.optimizer.maxEvaluations += number
            self.optimizer.learn()
        else:
            all_rewards = []
            all_ratios = []
            all_tp = []
            all_fix = []
            for dummy in range( number ):
                self.agent.newEpisode()
                rewards = []
                self.stepid = 0
                self.task.reset()
                print ' -->',
                while not self.task.isFinished():
                    r = self._oneInteraction()
                    rewards.append( r )
                all_rewards.append( rewards )
                all_ratios.append( self.task.ratio )
                all_tp.append( self.task.hasTarget )
                all_fix.append( self.task.nfix )
                print ' <-- %d %.2f' % ( self.task.nfix, sum( rewards ) )

            return all_rewards, all_ratios, all_tp, all_fix
