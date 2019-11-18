from collections import namedtuple
import json
from curious_agent.util.custom_json_encoder import CustomJsonEncoder
import logging

logger = logging.getLogger(__name__)


class DefaultMetaData(object):
    """
    Medata for model monitor and restore purpose
    """

    def __init__(self, fp, args):
        self.transition = namedtuple('Data',
                                     (
                                         "episode", "step", "time", "time_elapsed", "ep_len", "buffer_len", "epsilon",
                                         "reward",
                                         "avg_reward", "max_q", "max_avg_q",
                                         "loss", "avg_loss", "mode"))
        self.fp = fp
        self.data = None
        self.args = args

    def update(self, *args):
        """
        Update metadata
        :param args: args
        """
        self.data = self.transition(*args)
        if self.data.episode % self.args.disp_freq == 0:
            logger.info(
                f"E: {self.data.episode} | M: {self.data.buffer_len} |  Step: {self.data.step} | T: {self.data.time:.2f} | ET: {self.data.time_elapsed:.2f}"
                f" | Len: {self.data.ep_len} | EPS: {self.data.epsilon:.5f} | R: {self.data.reward} | AR: {self.data.avg_reward:.3f}"
                f" | MAQ:{self.data.max_avg_q:.2f} | L: {self.data.loss:.2f} | AL: {self.data.avg_loss:.4f} | Mode: {self.data.mode}")
        self.fp.write(self.data._asdict().values().__str__().replace('odict_values([', '').replace('])', '' + '\n'))

    def load(self, f):
        """
        Load Metadata
        :param f: File Pointer
        :return:
        """
        self.data = self.transition(*json.load(f).values())

    def dump(self, f):
        """
        JSONify metadata
        :param f: file pointer
        """
        json.dump(self.data._asdict(), f, cls=CustomJsonEncoder, indent=2)
