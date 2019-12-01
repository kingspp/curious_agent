from collections import namedtuple
import json
from curious_agent.util.custom_json_encoder import CustomJsonEncoder
import logging
from humanize import naturaltime
from curious_agent import MODULE_CONFIG


logger = logging.getLogger(__name__)


class A2CMetaData(object):
    """
    Medata for model monitor and restore purpose
    """

    def __init__(self, fp, args):
        self.episode_template = namedtuple('EpisodeData', ("episode", "time", "score", "loss"))
        self.fp = fp
        self.episode_data = None
        self.args = args
        if MODULE_CONFIG.BaseConfig.TENSORBOARD_SUMMARIES:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(MODULE_CONFIG.BaseConfig.PATH_GRAPHS)

    def update_episode(self, *args):
        """
        Update metadata
        :param args: args
        """
        self.episode_data = self.episode_template(*args)
        if self.episode_data.episode % self.args.disp_freq == 0:
            logger.info(
                f"E: {self.episode_data.episode} |  R: {self.episode_data.score} "
                f"T: {self.episode_data.time:.2f} |  L: {self.episode_data.loss:.2f} "
                )
        if MODULE_CONFIG.BaseConfig.TENSORBOARD_SUMMARIES:
            self.writer.add_scalar('episode/reward', self.episode_data.score, self.episode_data.episode)
            self.writer.add_scalar('episode/loss', self.episode_data.loss, self.episode_data.episode)

        self.fp.write(self.episode_data._asdict().values().__str__().replace('odict_values([', '').replace('])', '\n'))

    def load(self, f):
        """
        Load Metadata
        :param f: File Pointer
        :return:
        """
        self.episode_data = self.episode_data(*json.load(f).values())

    def dump(self, f):
        """
        JSONify metadata
        :param f: file pointer
        """
        json.dump(self.episode_data._asdict(), f, cls=CustomJsonEncoder, indent=2)
