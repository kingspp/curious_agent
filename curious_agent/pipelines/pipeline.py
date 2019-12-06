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
from curious_agent.environments.open_ai.atari.atari_environment import AtariEnvironment
from curious_agent.stats_recorders.open_ai_stats_recorders.atari_stats_recorder import AtariEnvStatsRecorder
from munch import Munch
import json
from curious_agent import MODULE_CONFIG, MODULE_CONFIG_DATA
from curious_agent.util import Directories, File
import os
from curious_agent.util import add_logs_to_tmp
import logging
import gc

logger = logging.getLogger(__name__)


class Pipeline(object):
    """Class defining the Pipeline that includes the commonalities that should govern all experiments run using this
    framework

    """

    @typechecked
    def __init__(self, train_agent: Agent,
                 environment: Environment,
                 config: Munch,
                 test: bool):
        # configuration
        self.name = config['pipeline_name']
        self.config = config
        # training elements
        self.train_agent = train_agent
        self.env = environment
        # testing elements
        self.probing_enabled = False

        # if both are provided, the testing functionality is enabled
        if test:
            self.probing_enabled = True

        # logging elements
        self.experiments_meta = {}
        self.current_experiments_meta = {"available_checkpoints": []}
        self.current_run = 1
        self.cur_exp_dir = ""
        logger.info("Config: " + json.dumps(self.config, indent=2))

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
                logger.info("Starting performance probing thread...")
                while not self._stop_event.is_set():

                    time.sleep(self.interval)
                    self.pipeline.check_performance()

        self.performanceProbingThread = PerformanceProbingThread(self, 20)

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
                if self.name in self.experiments_meta:
                    self.current_run = int(max(list(self.experiments_meta[self.name].keys()))) + 1
                    self.experiments_meta[self.name][self.current_run] = self.current_experiments_meta
                else:
                    self.experiments_meta[self.name] = {self.current_run: self.current_experiments_meta}
            else:
                self.experiments_meta = {self.name: {self.current_run: self.current_experiments_meta}}
        except:
            raise Exception("Existing experiments meta is corrupt.")

    @typechecked
    def create_experiments_dir(self, is_continuing: bool):
        # increment the number of runs for this experiment
        # update the in-memory json experiments' meta
        # self.experiments_meta[self.name][self.current_run] = {"available_checkpoints": []}
        # create the folder for the pipeline/experiment, if it's not there yet
        Directories.mkdir(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, self.name))
        # create the folder corresponding to the current run, and add the logs, checkpoints, graphs and configs
        # directories.
        self.cur_exp_dir = os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, self.name,
                                        str(self.current_run))
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
        if is_continuing:
            json.dump('{"is_continuing": 1}', open(f"{self.cur_exp_dir}/{path_configs}/continued_training_info.json",
                                                   'w'), indent=2)
        add_logs_to_tmp(path=os.path.join(self.cur_exp_dir, MODULE_CONFIG.BaseConfig.PATH_LOG))

    @typechecked
    def check_performance(self):
        """A method that crawls over the checkpoints and creates videos if needed

        :return: void
        """
        if type(self.env) == AtariEnvironment:
            pass
        else:
            raise NotImplementedError('No other StatsRecorder type is implemented.')
        checkpoints = []
        for fs_object in os.listdir(MODULE_CONFIG.BaseConfig.PATH_CHECKPOINT):
            # if os.path.isfile(fs_object):
            checkpoints.append(int(fs_object))
        if len(checkpoints) > 0:
            checkpoints.sort()
            logger.debug('Crawling over checkpoints. . .')
            for checkpoint in checkpoints:
                location = os.path.join(MODULE_CONFIG.BaseConfig.PATH_CHECKPOINT, str(checkpoint), "videos")
                if not os.path.exists(location):
                    # build a new test agent and environment
                    logger.debug('Running a test for checkpoint: ' + str(checkpoint))
                    test_env = type(self.env)(self.config['env_config'], atari_wrapper=True)
                    test_agent = type(self.train_agent)(test_env, self.config['agent_config'])
                    stats_recorder = AtariEnvStatsRecorder(test_agent, test_env, 30)

                    stats_recorder.load(location)
                    stats_recorder.record(location)
                    logger.debug('Finished the test for the checkpoint: ' + str(checkpoint))
                    gc.collect()
        logger.debug('Finished crawling. . .')

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
                logger.info(
                    f"Starting tensorboard @ http://localhost:{MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT}/#scalars")
            # start the training process (blocking)
            self.train_agent.train()
            if self.probing_enabled:
                # stop the performance probing thread
                self.performanceProbingThread.stop()
        except KeyboardInterrupt:
            self.cleanup()

    @typechecked
    def resume(self, run: int = -1, checkpoint: int = -1):
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
                logger.info(
                    f"Starting tensorboard @ http://localhost:{MODULE_CONFIG.BaseConfig.TENSORBOARD_PORT}/#scalars")
            # start the training process (blocking)
            self.train_agent.train(True)
            if self.probing_enabled:
                # stop the performance probing thread
                self.performanceProbingThread.stop()
        except KeyboardInterrupt:
            self.cleanup()

    def _load_pipeline(self, run: int = -1, checkpoint: int = -1):
        # Check if run exists
        if run == -1:
            run = str(max([int(i) for i in list(self.experiments_meta[self.name].keys())])-1)
        else:
            if str(run) not in self.experiments_meta[self.name]:
                raise Exception(f"Experiment run not found! Given: {run}")

        # Check if checkpoint exist
        if checkpoint == -1:
            if run in self.experiments_meta[self.name] and len(self.experiments_meta[self.name][run]['available_checkpoints']) > 0:
                checkpoint = str(max(self.experiments_meta[self.name][run]['available_checkpoints']))
            else:
                raise Exception(f"Checkpoints not available for the specified run. Given: Run: {run}")
        else:
            if checkpoint not in self.experiments_meta[self.name][run]['available_checkpoints']:
                raise Exception(f"Checkpoint for the given experiment not found: Give: {checkpoint}")

        # Get the path
        checkpoint_path =os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, self.name, run, MODULE_CONFIG.BaseConfig.PATH_CHECKPOINT, checkpoint, f"e_{checkpoint}")
        self.train_agent.load(file_name_with_path=checkpoint_path)

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
