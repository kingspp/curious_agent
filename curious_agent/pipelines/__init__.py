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


class Pipeline(object):
    """Class defining shared functionality that environments have to implement to implement specific RL algorithms'
    logic

    The train function below contains essential boilerplate that needs to be used in the extensions of this class. The
    This boilerplate exemplifies the idiomatic restrictions that should be followed to preserve the reliability of the
    training process and experiment collection.

    The other function that needs implementation is "test". This function will be called by a thread that is launched by
    by the generic init function.

    """

    def __init__(self):
        self.state = []  # TODO: replace with a munch object of the same name
        self.models = []  # TODO: replace with a munch object of the same name

        class PerformanceProbingThread(threading.Thread):
            """Class defining the behavior of the performance probing thread

            This thread keeps calling the "test" function at an interval determined in the constructor

            """
            @typechecked
            def __init__(self, interval: int):
                """ Performance probing thread constructor

                :param interval: the interval at which the "test" function will be called
                """
                threading.Thread.__init__(self)
                self.interval = interval
                self._stop_event = threading.Event()

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
                    self.test()

        # launch the performance probing thread
        self.performanceProbingThread = PerformanceProbingThread(10)
        self.performanceProbingThread.start()

    @abstractmethod
    @typechecked
    def test(self):
        """A method that tests the latest model output during an experiment and saves statistics and video footage of
        the test run

        This test function is specific for each pipeline and needs to be implemented

        :return: void
        """
        raise NotImplementedError

    # TODO: needs generic implementation
    @typechecked
    def load(self):
        """A method that loads the state variables and the models to enable the experiment to continue from a halted
        state

        This method affect the self.state and self.models variables.

        :return: void
        """
        pass

    # TODO: needs generic implementation
    @typechecked
    def save(self):
        """A method that saves the models and state variables to enable the safe halting and resumption of experiments

        This method's side effect is only restricted to the file-system.

        :return: void
        """
        pass

    @abstractmethod
    @typechecked
    def train(self, continuing: bool):
        """A method that contains the whole reinforcement learning algorithm

        The implementations of this method should respect the following idiomatic restrictions to ensure reliable
        training experiments:
            - Include two initialization branches: "starting" and "continuing", each referring to the cases of starting
            off, respectively, from a first-time-run or a paused state. This can be done using an if-else statement at
            the beginning of the train function, with each branch containing the corresponding logic.
            - Program assuming the "continuing logic", which means that at any point in the program outside of the
            initialization branches, we assume as if the program is continuing. This can be done, for example, by
            avoiding for loops that start from episode 0, and instead using ones that start from the current episode, of
             course after initializing the episode number in the "continuing" branch of the initialization.
            - besides that, the body can contain whatever code construct that is needed for the algorithm to be
            implemented, like for-loops, while-loops, if-else statements, nested for-loops etc...

        :param continuing: a boolean indicating if the algorithm is continuing, so that it can tell which initialization
        branch to use.

        :return: None
        """

        # the initialization branches mentioned above
        if not continuing:  # if starting from scratch
            pass  # custom startup initialization
        else:  # if resuming training
            self.load()  # loads the state variable which contains everything you need

        # after the initialization branches, the function should be implemented as if the algorithm is continuing from
        # a halted state, as well as from a startup-state. the same logic should work, in either case.

        # finally, stop the performance probing thread
        self.performanceProbingThread.stop()
        raise NotImplementedError
