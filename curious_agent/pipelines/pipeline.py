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
from munch import Munch
import json
from curious_agent import MODULE_CONFIG, MODULE_CONFIG_DATA
from curious_agent.util import Directories, File
import os
from curious_agent.util import add_logs_to_tmp
import logging

logger = logging.getLogger(__name__)


class Pipeline(object):
    """Class defining shared functionality that environments have to implement to implement specific RL algorithms'
    logic

    The train function below contains essential boilerplate that needs to be used in the extensions of this class. The
    This boilerplate exemplifies the idiomatic restrictions that should be followed to preserve the reliability of the
    training process and experiment collection.

    The other function that needs implementation is "test". This function will be called by a thread that is launched by
    by the generic init function.

    """

    def __init__(self, train_agent: Agent,
                 environment: Environment,
                 config: Munch,
                 test_agent: Agent = None, test_environment: Environment = None):
        self.name = config['pipeline_name']
        self.config = config
        self.train_agent = train_agent
        self.test_environment = test_environment
        # different agent and environment references to avoid data-races
        self.stats_recorder = StatsRecorder(test_agent, test_environment)
        # make sure that we do not let the environment be used by the agent.
        # assert self.agent.env != self.test_environment, "The agent and environment in the pipeline's arguments " \
        #                                                 "should not be related. Use a new instance of the environment."
        # # make sure that the train agent and test agent are of the same type
        # assert isinstance(test_agent, train_agent), "The train and test agent should be of the same type."
        self.experiments_meta = {"experiments": {}}
        self.current_experiments_meta = {"runs": 0}
        self.cur_exp_dir = ""
        if not MODULE_CONFIG.BaseConfig.DRY_RUN:
            self.create_base_directories()

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

    def create_base_directories(self):
        Directories.mkdir(MODULE_CONFIG.BaseConfig.BASE_DIR)
        meta_file = MODULE_CONFIG.BaseConfig.BASE_DIR + "/" + MODULE_CONFIG.BaseConfig.EXPERIMENTS_META_NAME + '.json'
        try:
            if File.is_exist(meta_file):
                self.experiments_meta = json.load(open(meta_file))
                if self.name in self.experiments_meta['experiments']:
                    self.current_experiments_meta = self.experiments_meta['experiments'][self.name]
        except:
            pass

    def create_experiments_dir(self):
        self.current_experiments_meta['runs'] += 1
        self.experiments_meta['experiments'][self.name] = self.current_experiments_meta
        Directories.mkdir(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, self.name))
        self.cur_exp_dir = os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, self.name,
                                        str(self.current_experiments_meta['runs']))
        MODULE_CONFIG.BaseConfig.PATH_LOG = os.path.join(self.cur_exp_dir, MODULE_CONFIG.BaseConfig.PATH_LOG)
        MODULE_CONFIG.BaseConfig.PATH_CHECKPOINT = os.path.join(self.cur_exp_dir,
                                                                MODULE_CONFIG.BaseConfig.PATH_CHECKPOINT)
        MODULE_CONFIG.BaseConfig.PATH_GRAPHS = os.path.join(self.cur_exp_dir, MODULE_CONFIG.BaseConfig.PATH_GRAPHS)
        MODULE_CONFIG.BaseConfig.PATH_CONFIGS = os.path.join(self.cur_exp_dir, MODULE_CONFIG.BaseConfig.PATH_CONFIGS)
        Directories.mkdir(self.cur_exp_dir)
        Directories.mkdir(MODULE_CONFIG.BaseConfig.PATH_LOG)
        Directories.mkdir(MODULE_CONFIG.BaseConfig.PATH_CHECKPOINT)
        Directories.mkdir(MODULE_CONFIG.BaseConfig.PATH_GRAPHS)
        Directories.mkdir(MODULE_CONFIG.BaseConfig.PATH_CONFIGS)
        json.dump(MODULE_CONFIG_DATA, open(f"{self.cur_exp_dir}/configs/base_config.json", 'w'), indent=2)
        json.dump(self.config, open(f"{self.cur_exp_dir}/configs/pipeline_config.json", 'w'), indent=2)
        json.dump(self.experiments_meta, open(
            MODULE_CONFIG.BaseConfig.BASE_DIR + "/" + MODULE_CONFIG.BaseConfig.EXPERIMENTS_META_NAME + '.json', 'w'),
                  indent=2)
        add_logs_to_tmp(path=os.path.join(self.cur_exp_dir, MODULE_CONFIG.BaseConfig.PATH_LOG))

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
        if not MODULE_CONFIG.BaseConfig.DRY_RUN:
            self.create_experiments_dir()
        try:
            self.performanceProbingThread.start()
            # start the training process (blocking)
            os.system(
                f"tensorboard --logdir {MODULE_CONFIG.BaseConfig.PATH_GRAPHS} --port {MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT} > {MODULE_CONFIG.BaseConfig.PATH_LOG}/tb.log 2>&1 &")
            logger.info(f"Starting tensorboard @ http://localhost:{MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT}/#scalars")
            self.train_agent.train(False)
            # stop the performance probing thread
            self.performanceProbingThread.stop()
        except KeyboardInterrupt:
            self.cleanup()

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
        try:
            self.performanceProbingThread.start()
            # start the training process (blocking)
            os.system(
                f"tensorboard --logdir {MODULE_CONFIG.BaseConfig.PATH_GRAPHS} --port {MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT}")
            logger.info(f"Starting tensorboard @ http://localhost:{MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT}/#scalars")
            self.train_agent.train(True)
            # stop the performance probing thread
            self.performanceProbingThread.stop()
        except KeyboardInterrupt:
            self.cleanup()

    def cleanup(self):
        os.system(f"kill -9 $(lsof -t -i:{MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT} -sTCP:LISTEN)")

    def collect_garbage(self):
        """
        Collect garbage based on condition
        :param i_episode: Episode Number
        """
        # if i_episode % MODULE_CONFIG.BaseConfig.GC_FREQUENCY == 0:
        #     print("Executing garbage collector . . .")
        #     gc.collect()
        raise NotImplementedError
