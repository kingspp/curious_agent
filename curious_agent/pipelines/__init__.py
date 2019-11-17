# -*- coding: utf-8 -*-
"""
.. module:: pipelines
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
    def __init__(self, agent: Agent, test_environment: Environment, stat_recorder: StatRecorder):
        self.agent = agent
        self.test_environment = test_environment
        self.stat_recorder = stat_recorder
        # make sure that we do not let the environment be used by the agent.
        assert self.agent.env != self.test_environment, "The agent and environment in the pipeline's arguments " \
                                                        "should not be related. Use a new instance of the environment."

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
        self.stat_recorder.record(self.agent, self.test_environment)
        raise NotImplementedError

    # @typechecked
    # def load(self):
    #     """A method that loads the state variables and the models to enable the experiment to continue from a halted
    #     state
    #
    #     This method affect the self.state and self.models variables.
    #
    #     :return: void
    #     """
    #     # TODO: needs generic implementation
    #     pass
    #
    # @typechecked
    # def save(self):
    #     """A method that saves the models and state variables to enable the safe halting and resumption of experiments
    #
    #     This method's side effect is only restricted to the file-system.
    #
    #     :return: void
    #     """
    #     # TODO: needs generic implementation
    #     pass

    @typechecked
    def execute(self):
        """Method that starts the training and handles the performance thread correctly

        :return: void
        """
        # TODO: setup the directory structure that corresponds to this experiment
        # launch the performance probing thread
        self.performanceProbingThread.start()
        # start the training process (blocking)
        self.agent.train(False)
        # stop the performance probing thread
        self.performanceProbingThread.stop()

    @typechecked
    def resume(self, checkpoint: int):
        """Methods that continues the training from previous checkpoints

        :param checkpoint: the index of the checkpoint in reverse order, starting from the last checkpoint. 0 is the
        last one, 1 is the checkpoint before, etc...

        :note: if there are no checkpoints, the method simply returns.

        :return: void
        """
        # TODO: reuse the directory structure that corresponds to this experiment
        # launch the performance probing thread
        self.performanceProbingThread.start()
        self.agent.load(checkpoint)
        # start the training process (blocking)
        self.agent.train(True)
        # stop the performance probing thread
        self.performanceProbingThread.stop()

    # @abstractmethod
    # @typechecked
    # def train(self, continuing: bool):
    #     """A method that contains the whole reinforcement learning algorithm
    #
    #     The implementations of this method should respect the following idiomatic restrictions to ensure reliable
    #     training experiments:
    #         - Include two initialization branches: "starting" and "continuing", each referring to the cases of starting
    #         off, respectively, from a first-time-run or a paused state. This can be done using an if-else statement at
    #         the beginning of the train function, with each branch containing the corresponding logic.
    #         - Program assuming the "continuing logic", which means that at any point in the program outside of the
    #         initialization branches, we assume as if the program is continuing. This can be done, for example, by
    #         avoiding for loops that start from episode 0, and instead using ones that start from the current episode, of
    #          course after initializing the episode number in the "continuing" branch of the initialization.
    #         - besides that, the body can contain whatever code construct that is needed for the algorithm to be
    #         implemented, like for-loops, while-loops, if-else statements, nested for-loops etc...
    #
    #     :param continuing: a boolean indicating if the algorithm is continuing, so that it can tell which initialization
    #     branch to use.
    #
    #     :return: None
    #     """
    #
    #     # the initialization branches mentioned above
    #     if not continuing:  # Starting
    #         pass  # custom startup initialization
    #     else:  # Continuing
    #         pass  # custom continuing initialization
    #
    #     # after the initialization branches, the function should be implemented as if the algorithm is continuing from
    #     # a halted state, as well as from a startup-state. the same logic should work, in either case.
    #     raise NotImplementedError
