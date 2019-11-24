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
    """Class defining the Pipeline that includes the commonalities that should govern all experiments run using this
    framework

    """

    @typechecked
    def __init__(self, train_agent: Agent,
                 environment: Environment,
                 config: Munch,
                 test_agent: Agent = None, test_environment: Environment = None):
        # configuration
        self.name = config['pipeline_name']
        self.config = config
        # training elements
        self.train_agent = train_agent
        self.env = environment
        # testing elements
        self.test_environment = test_environment
        self.probing_enabled = False

        # if both are provided, the testing functionality is enabled
        if test_agent is not None and test_environment is not None:
            self.probing_enabled = True
            # make sure that the train agent and test agent are of the same type
            assert isinstance(test_agent, type(train_agent)), "The train and test agent should be of the same type."
            self.stats_recorder = StatsRecorder(test_agent, test_environment)
            # make sure that we do not let the environment be used by the agent.
            assert self.train_agent.env != self.test_environment, "The agent and environment in the pipeline's "\
                                                                  "arguments should not be related. Use a new "\
                                                                  "instance of the environment."
        # logging elements
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

    @typechecked
    def create_base_directories(self):
        """Method that creates the experiments file if the base_dir is not set in the base_config file, or the custom
        one specified otherwise.

        This function also loads the current experiment's meta from that file if it is already there. It does nothing,
        otherwise.

        :return: void
        """
        Directories.mkdir(MODULE_CONFIG.BaseConfig.BASE_DIR)
        meta_file = MODULE_CONFIG.BaseConfig.BASE_DIR + "/" + MODULE_CONFIG.BaseConfig.EXPERIMENTS_META_NAME + '.json'
        try:
            if File.is_exist(meta_file):
                self.experiments_meta = json.load(open(meta_file))
                if self.name in self.experiments_meta['experiments']:
                    self.current_experiments_meta = self.experiments_meta['experiments'][self.name]
        except:
            pass

    @typechecked
    def create_experiments_dir(self, is_continuing: bool):
        # increment the number of runs for this experiment
        self.current_experiments_meta['runs'] += 1
        # update the in-memory json experiments' meta
        self.experiments_meta['experiments'][self.name] = self.current_experiments_meta
        # create the folder for the pipeline/experiment, if it's not there yet
        Directories.mkdir(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, self.name))
        # create the folder corresponding to the current run, and add the logs, checkpoints, graphs and configs
        # directories.
        self.cur_exp_dir = os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, self.name,
                                        str(self.current_experiments_meta['runs']))
        json.dump(self.experiments_meta, open(
            MODULE_CONFIG.BaseConfig.BASE_DIR + "/" + MODULE_CONFIG.BaseConfig.EXPERIMENTS_META_NAME + '.json', 'w'),
                  indent=2)
        MODULE_CONFIG.BaseConfig.BASE_DIR = self.cur_exp_dir
        path_configs = MODULE_CONFIG.BaseConfig.PATH_CONFIGS
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
        # write the configurations of that run in its configurations folder, for retrospective inspections
        json.dump(MODULE_CONFIG_DATA, open(f"{self.cur_exp_dir}/{path_configs}/base_config.json", 'w'), indent=2)
        json.dump(self.config, open(f"{self.cur_exp_dir}/{path_configs}/pipeline_config.json",
                                    'w'), indent=2)
        # write the experiments json that keeps track of each experiment's number of runs, and log
        json.dump(self.experiments_meta, open(
            MODULE_CONFIG.BaseConfig.BASE_DIR + "/" + MODULE_CONFIG.BaseConfig.EXPERIMENTS_META_NAME + '.json', 'w'),
                  indent=2)
        if is_continuing:
            json.dump('{"is_continuing": 1}', open(f"{self.cur_exp_dir}/{path_configs}/continued_training_info.json",
                                                   'w'), indent=2)
        add_logs_to_tmp(path=os.path.join(self.cur_exp_dir, MODULE_CONFIG.BaseConfig.PATH_LOG))

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
        """Method that starts the training and handles the performance thread correctly, if enabled

        It also creates a new folder tree-structure storing the run's information

        :return: void
        """
        if not MODULE_CONFIG.BaseConfig.DRY_RUN:
            self.create_experiments_dir(is_continuing=False)
        # run = -2
        # checkpoint = 1
        # runs = []
        # for fs_object in os.listdir(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, self.name)):
        #     # if os.path.isfile(fs_object):
        #     runs.append(int(fs_object))
        # # validate that the index is not out of bounds
        # if run >= len(runs):
        #     raise ValueError("The checkpoint has to be from 0 to " + str(len(list) - 1))
        # runs.sort()
        # run = str(runs[run])
        # print(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, self.name, run, MODULE_CONFIG.BaseConfig.PATH_CHECKPOINT, str(checkpoint)))
        #
        # exit()
        try:
            if self.probing_enabled:
                self.performanceProbingThread.start()
            if not MODULE_CONFIG.BaseConfig.DRY_RUN:
                # create file structure for TensorBoardX logging
                os.system(
                    f"tensorboard --logdir {MODULE_CONFIG.BaseConfig.PATH_GRAPHS} --port {MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT} > {MODULE_CONFIG.BaseConfig.PATH_LOG}/tb.log 2>&1 &")
                logger.info(f"Starting tensorboard @ http://localhost:{MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT}/#scalars")
            # start the training process (blocking)
            self.train_agent.train(False)
            if self.probing_enabled:
                # stop the performance probing thread
                self.performanceProbingThread.stop()
        except KeyboardInterrupt:
            self.cleanup()

    @typechecked
    def resume(self, run: int=-1, checkpoint: int=-1):
        """Methods that continues the training from previous checkpoints. Everything stored in the recommended ways for
        hot reloads in this framework is restored for training resumption

        :param run: the index of the run to reload from. i.e. 0 is for the first checkpoint, -1 for the last one, etc...
        :param checkpoint: the index of the checkpoint to reload from. i.e. 0 is for the first checkpoint, -1 for the
        last one, etc...

        :note: if there are no checkpoints, the method simply returns.

        :return: void
        """
        # TODO 2: resolve the name of the checkpoint folder and get the agent's (directory operations)
        if not MODULE_CONFIG.BaseConfig.DRY_RUN:
            self.create_experiments_dir(is_continuing=True)


        # for fs_object in os.listdir(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, self.name)):
        #     if os.path.isdir(fs_object):
        #         pass
        # TODO 3: if destructive, remove the checkpoints that come after the one that was loaded
        # launch the performance probing thread
        try:
            if self.probing_enabled:
                self.performanceProbingThread.start()
            if not MODULE_CONFIG.BaseConfig.DRY_RUN:
                # create file structure for TensorBoardX logging
                os.system(
                    f"tensorboard --logdir {MODULE_CONFIG.BaseConfig.PATH_GRAPHS} --port {MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT} > {MODULE_CONFIG.BaseConfig.PATH_LOG}/tb.log 2>&1 &")
                logger.info(f"Starting tensorboard @ http://localhost:{MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT}/#scalars")
            # start the training process (blocking)
            self.train_agent.train(True)
            if self.probing_enabled:
                # stop the performance probing thread
                self.performanceProbingThread.stop()
        except KeyboardInterrupt:
            self.cleanup()

    @typechecked
    def cleanup(self):
        os.system(f"kill -9 $(lsof -t -i:{MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT} -sTCP:LISTEN)")

    @abstractmethod
    @typechecked
    def collect_garbage(self):
        """
        Collect garbage based on condition
        :param i_episode: Episode Number
        """
        # if i_episode % MODULE_CONFIG.BaseConfig.GC_FREQUENCY == 0:
        #     print("Executing garbage collector . . .")
        #     gc.collect()
        raise NotImplementedError
