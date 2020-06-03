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
from curious_agent.util import Directories

logger = logging.getLogger(__name__)


class Agent(metaclass=ABCMeta):

    @typechecked
    def __init__(self, env: Environment, agent_config: dict):
        self.env = env
        self.state = Munch({"config": agent_config})
        self.state._models = None
        # self.meta = DefaultMetaData

    @abstractmethod
    def take_action(self, observation: np.array, test: bool = True, **args):
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
    @typechecked
    def train(self, persist: bool, run: int = -1, checkpoint: int = -1):
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

    def register_models(self):
        self.state._models = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Model):
                v.name = k if v.name == "" else v.name
                self.state._models[v.name] = ref(v)()


    @typechecked
    def save(self, i_episode):
        """Method that saves the models and state variables to enable the safe halting and resumption of experiments

        This method's side effect is only restricted to the file-system.

        :note: notice how this is not an abstract function; you don't need to implement it

        :param path: the name of the file on the disk (ideally relative to the working directory)

        :return: void
        """
        if i_episode % self.state.config.save_freq == 0:
            if self.state._models is None:
                self.register_models()
            save_dir = os.path.join(MODULE_CONFIG.BaseConfig.PATH_CHECKPOINT, str(i_episode))
            Directories.mkdir(save_dir)
            for k, model in self.state._models.items():
                model.save(
                    file_name_with_path=os.path.join(save_dir,
                                                     f'e_{i_episode}_{k if model.name == "" else model.name}.th'))

            with open(os.path.join(save_dir, f"e_{i_episode}.meta"), 'w') as f:
                json.dump(self.state, f, cls=CustomJsonEncoder, indent=2)
            _exp_meta = json.load(open(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, '..', '..',
                                                    MODULE_CONFIG.BaseConfig.EXPERIMENTS_META_NAME + '.json')))
            _exp_name = MODULE_CONFIG.BaseConfig.BASE_DIR.split('/')[-2]
            _exp_run = MODULE_CONFIG.BaseConfig.BASE_DIR.split('/')[-1]
            _exp_meta[_exp_name][_exp_run]['available_checkpoints'].append(i_episode)
            json.dump(_exp_meta, open(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, '..', '..',
                                                   MODULE_CONFIG.BaseConfig.EXPERIMENTS_META_NAME + '.json'), 'w'),
                      indent=2)

    @typechecked
    def load(self, file_name_with_path: str):
        """Method that saves the models and state variables to enable the safe halting and resumption of experiments

        This method's side effect is only restricted to the file-system.

        :note: notice how this is not an abstract function; you don't need to implement it

        :param file_name_with_path: the name of the file on the disk (ideally relative to the working directory)

        :return: void
        """

        if self.state._models is None:
            self.register_models()
        logger.info("Agent State loaded successfully")
        for k, model in self.state._models.items():
            model.load(file_name_with_path=os.path.join(f'{file_name_with_path}_{model.name}.th'))
            logger.info(f'{file_name_with_path}_{model.name}.th loaded')
            logger.info(f"{model.name} model loaded successfully")
        self.state = Munch(json.load(open(file_name_with_path + ".meta")))
