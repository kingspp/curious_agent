"""

### NOTICE ###
DO NOT revise this file

"""
from curious_agent.environments.environment import Environment
from abc import ABCMeta, abstractmethod
from munch import Munch
from typeguard import typechecked
import numpy as np
import os
import logging
import torch
from curious_agent.models import Model
from weakref import ref
from curious_agent import MODULE_CONFIG
import json
from curious_agent.util import CustomJsonEncoder
from curious_agent.meta.default_meta import DefaultMetaData

logger = logging.getLogger(__name__)


# def register_models(f, always: bool = False):
#     """
#     | **@author:** Prathyush SP
#     |
#     | Key Exception Decorator.
#     :param f: Function
#     :return: Function Return Parameter
#     """
#
#     if f is None:
#         return partial(keyexception, always=always)
#
#     @wraps(f)
#     def wrapped(*args, **kwargs):
#         try:
#             r = f(*args, **kwargs)
#             return r
#         except KeyError as ke:
#             # _matches = re.findall(r'\[(.*?)\]', message.__str__())
#             #
#             # for k, v in message.items():
#             #     print(k)
#             # print(_matches)
#
#             # _message = {_matches[k]: v for k,v  in message.items() if k in _matches}
#             # print(_message, _matches[0], message.__str__())
#             # exit()
#             # # _message = message.format(**{k:v for k,v in kwargs.items()  if k in _matches}) if len(_matches) > 0 else message
#             # import traceback
#             # print('keys:', ke)
#             # print(traceback.format_exc())
#             #
#             # if component_name and message:
#             #     logger.error(_message)
#             #     raise RztdlException(component_name=component_name, message=_message, errors=ke)
#             # else:
#             #     logger.error('Key Error - Attribute: {} Value:{}'.format(
#             #         [k for k, v in kwargs.items() if v == ke.__str__().strip().replace('\'', '')], ke))
#             raise ke
#
#     return wrapped

class Agent(metaclass=ABCMeta):

    @typechecked
    def __init__(self, env: Environment, agent_config: dict):
        self.env = env
        self.state = Munch({"config": agent_config})
        self._models = None
        self.meta = DefaultMetaData

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
    def train(self, persist: bool):
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

        :param persist: a boolean indicating if the algorithm is continuing, so that it can tell which initialization
        branch to use.

        :return: None
        """

        # the initialization branches mentioned above
        if not persist:  # Starting
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

        This method affects the self.state.config and self.models variables.

        :note: notice how this is not an abstract function; you don't need to implement it

        :param checkpoint_folder: the location of the checkpoint folder on disk (ideally relative to the working
        directory)
        :return: void
        """
        raise NotImplementedError

    def register_models(self):
        self._models = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Model):
                self._models[k if v.name == "" else v.name] = ref(v)()

    @typechecked
    def save(self, i_episode):
        """Method that saves the models and state variables to enable the safe halting and resumption of experiments

        This method's side effect is only restricted to the file-system.

        :note: notice how this is not an abstract function; you don't need to implement it

        :return: void
        """
        if i_episode % self.state.config.save_freq == 0:
            if self._models is None:
                self.register_models()
            for k, model in self._models.items():
                model.save(
                    file_name_with_path=os.path.join(MODULE_CONFIG.BaseConfig.PATH_CHECKPOINT,
                                                     f'e_{i_episode}_{k if model.name == "" else model.name}.th'))

            with open(os.path.join(MODULE_CONFIG.BaseConfig.PATH_CHECKPOINT, f"e_{i_episode}.meta"), 'w') as f:
                json.dump(self.state, f, cls=CustomJsonEncoder)
