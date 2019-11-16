from abc import ABCMeta, abstractmethod
from curious_agent.models import Model
import typing
from typeguard import typechecked
from munch import Munch
from curious_agent import MODULE_CONFIG
import gc


class Pipeline(metaclass=ABCMeta):
    def __init__(self):
        self.state = Munch()

    @typechecked
    def save_model(self, models: typing.List[Model]):
        pass

    def load_model(self):
        pass

    def collect_garbage(self, i_episode):
        """
        Collect garbage based on condition
        :param i_episode: Episode Number
        """
        if i_episode % MODULE_CONFIG.BaseConfig.GC_FREQUENCY == 0:
            print("Executing garbage collector . . .")
            gc.collect()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
