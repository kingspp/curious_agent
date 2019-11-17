# -*- coding: utf-8 -*-
"""
.. module:: pipeline
    :platform: Linux
    :synopsis: module that contains the definition of the abstract pipeline forms the basis of different learning
    algorithm implementations

.. moduleauthor:: Sinan Morcel <shmorcel@wpi.edu>
"""

from abc import abstractmethod
from typeguard import typechecked
import threading
import time
from curious_agent.agents import Agent
from curious_agent.environments import Environment
from curious_agent.stats_recorders.stats_recorder import StatsRecorder


class Pipeline(object):
    """Class defining shared functionality that environments have to implement to implement specific RL algorithms'
    logic

    The train function below contains essential boilerplate that needs to be used in the extensions of this class. The
    This boilerplate exemplifies the idiomatic restrictions that should be followed to preserve the reliability of the
    training process and experiment collection.

    The other function that needs implementation is "test". This function will be called by a thread that is launched by
    by the generic init function.

    """

    @typechecked
    def __init__(self, train_agent: Agent, test_agent: Agent, test_environment: Environment):
        self.train_agent = train_agent
        self.test_environment = test_environment
        # different agent and environment references to avoid data-races
        self.stats_recorder = StatsRecorder(test_agent, test_environment)
        # make sure that we do not let the environment be used by the agent.
        assert self.agent.env != self.test_environment, "The agent and environment in the pipeline's arguments " \
                                                        "should not be related. Use a new instance of the environment."
        # make sure that the train agent and test agent are of the same type
        assert isinstance(test_agent, train_agent), "The train and test agent should be of the same type."

        class PerformanceProbingThread(threading.Thread):
            """Class defining the behavior of the performance probing thread

            This thread keeps calling the "test" function at an interval determined in the constructor

            """
            @typechecked
            def __init__(self, pipeline: Pipeline, interval: int):
                """ Performance probing thread constructor

                :param interval: the interval at which the "test" function will be called
                """
                threading.Thread.__init__(self)
                self.interval = interval
                self._stop_event = threading.Event()
                self.pipeline = pipeline

            @typechecked
            def stop(self):
                self._stop_event.set()

            @typechecked
            def stopped(self):
                return self._stop_event.is_set()

            @typechecked
            def run(self):
                """A function that runs the "test" function at a fixed interval determined in the constructor

                :return: void
                """
                while not self.stopped:
                    time.sleep(self.interval)
                    self.pipeline.performance_stats()

        self.performanceProbingThread = PerformanceProbingThread(self, 10)

    @abstractmethod
    @typechecked
    def performance_stats(self):
        """A method that tests the latest model output during an experiment and saves statistics and video footage of
        the test run

        This test assumes that the agent and the test_environment are compatible, if they are not, an error may result

        :return: void
        """
        # TODO: resolve the location of the stats using the info: user, experiment# and checkpoint# available on the
        #  file system (directory operations)
        location = "stats"
        self.stats_recorder.load(location + "name of checkpoint folder or weights")
        self.stats_recorder.record(location + "output")

    @typechecked
    def execute(self):
        """Method that starts the training and handles the performance thread correctly

        :return: void
        """
        # TODO: setup the directory structure that corresponds to this experiment
        # launch the performance probing thread
        self.performanceProbingThread.start()
        # start the training process (blocking)
        self.train_agent.train(False)
        # stop the performance probing thread
        self.performanceProbingThread.stop()

    @typechecked
    def resume(self, checkpoint: int, destructive: bool):
        """Methods that continues the training from previous checkpoints

        :param checkpoint: the index of the checkpoint in reverse order, starting from the last checkpoint. 0 is the
        last one, 1 is the checkpoint before, etc...
        :param destructive: bool that indicates if the future checkpoints beyond the one that gets loaded are removed

        :note: if there are no checkpoints, the method simply returns.

        :return: void
        """
        # TODO 1: reuse the directory structure that corresponds to this experiment
        # TODO 2: resolve the name of the checkpoint folder and get the agent's (directory operations)
        # TODO 3: if destructive, remove the checkpoints that come after the one that was loaded
        # launch the performance probing thread
        self.performanceProbingThread.start()
        # start the training process (blocking)
        self.train_agent.train(True)
        # stop the performance probing thread
        self.performanceProbingThread.stop()

    def collect_garbage(self):
        """
        Collect garbage based on condition
        :param i_episode: Episode Number
        """
        # if i_episode % MODULE_CONFIG.BaseConfig.GC_FREQUENCY == 0:
        #     print("Executing garbage collector . . .")
        #     gc.collect()
        raise NotImplementedError
