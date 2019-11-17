"""

### NOTICE ###
DO NOT revise this file

"""
from curious_agent.environments import Environment
from abc import abstractmethod
from typeguard import typechecked
from numpy import np


class Agent(object):
    @typechecked
    def __init__(self, env: Environment):
        self.env = env

    @abstractmethod
    def make_action(self, observation: np.array, test: bool = True):
        """Method that returns predicted action of the agent

        There is an idiomatic restriction that should go into the implementation of this function: the pre-processing
        of the state has to be done at the beginning of this function. Distinguishing between test and non-test is not
        recommended, and is actually discouraged when considering only this function. You should instead, use the train
        function to decrement the epsilon, for example... The boolean argument however was left there for backward
        compatibility and experience reuse.

        :param observation: np.array that contains the observed data
        :param test: boolean to indicate if the action is for training or testing

        :return action: int that encodes the predicted action from the trained model
        """
        raise NotImplementedError

    @abstractmethod
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        raise NotImplementedError

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
        if not continuing:  # Starting
            pass  # custom startup initialization
        else:  # Continuing
            pass  # custom continuing initialization

        # after the initialization branches, the function should be implemented as if the algorithm is continuing from
        # a halted state, as well as from a startup-state. the same logic should work, in either case.
        raise NotImplementedError

    @typechecked
    def load(self, checkpoint_folder: str):
        """Method that loads the state variables and the models to enable the experiment to continue from a halted
        state

        This method affects the self.state and self.models variables.

        :note: notice how this is not an abstract function; you don't need to implement it

        :param checkpoint_folder: the location of the checkpoint folder on disk (ideally relative to the working
        directory)
        :return: void
        """
        raise NotImplementedError

    @typechecked
    def save(self):
        """Method that saves the models and state variables to enable the safe halting and resumption of experiments

        This method's side effect is only restricted to the file-system.

        :note: notice how this is not an abstract function; you don't need to implement it

        :return: void
        """
        raise NotImplementedError
