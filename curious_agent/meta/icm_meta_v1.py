from collections import namedtuple
import json
from curious_agent.util.custom_json_encoder import CustomJsonEncoder
import logging
from humanize import naturaltime
from curious_agent import MODULE_CONFIG

logger = logging.getLogger(__name__)


class ICMMetaDataV1(object):
    """
    Medata for model monitor and restore purpose
    """

    def __init__(self, fp, args):
        self.episode_template = namedtuple('EpisodeData',
                                           ('episode', 'steps', 'epsilon', 'reward',
                                            'avg_reward', 'loss', 'intrinsic_reward', 'avg_intrinsic_reward', 'mode'))
        self.fp = fp
        self.episode_data = None
        self.step_data = None
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
                f"E: {self.episode_data.episode} |  Step: {self.episode_data.steps} "                
                f"| EPS: {self.episode_data.epsilon:.5f} "
                f"| R: {self.episode_data.reward} | AR: {self.episode_data.avg_reward:.3f} "
                f"| L: {self.episode_data.loss:.2f} | INTR: {self.episode_data.intrinsic_reward:.2f} "
                f"| AVG_INTR: {self.episode_data.intrinsic_reward:.2f} "
                f"| Mode: {self.episode_data.mode}")
        if MODULE_CONFIG.BaseConfig.TENSORBOARD_SUMMARIES:
            self.writer.add_scalar('episode/epsilon', self.episode_data.epsilon, self.episode_data.episode)
            self.writer.add_scalar('episode/steps', self.episode_data.steps, self.episode_data.episode)
            self.writer.add_scalar('episode/reward', self.episode_data.reward, self.episode_data.episode)
            self.writer.add_scalar('episode/avg_reward', self.episode_data.avg_reward, self.episode_data.episode)
            self.writer.add_scalar('episode/loss', self.episode_data.loss, self.episode_data.episode)
            self.writer.add_scalar('episode/intr_reward', self.episode_data.intrinsic_reward, self.episode_data.episode)
            self.writer.add_scalar('episode/avg_intr_reward', self.episode_data.avg_intrinsic_reward,
                                   self.episode_data.episode)

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
