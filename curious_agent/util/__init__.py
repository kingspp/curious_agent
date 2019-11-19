from curious_agent.util.custom_json_encoder import CustomJsonEncoder
import json
from munch import Munch
import os
import logging
import shutil
from typing import List, Union, Dict
import subprocess
from collections import Counter, OrderedDict
from datetime import datetime
from curious_agent import MODULE_CONFIG
import yaml
import uuid

logger = logging.getLogger(__name__)


def pipeline_config_loader(config_file_path):
    data = json.load(open(config_file_path))
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = Munch(v)
    return Munch(data)


class Directories:
    """
    | **@author:** Prathyush SP
    |
    | Directories Class
    """

    @staticmethod
    def is_exist(path: str) -> bool:
        """
        | **@author:** Prathyush SP
        |
        | Check if the directory Exists
        :param path: Directory path
        :return: Bool
        """
        return os.path.isdir(path)

    @staticmethod
    def is_empty(path: str) -> bool:
        """
        | **@author:** Prathyush SP
        |
        | Check if the directory is empty
        :param path: Directory path
        :return: Bool
        """
        if Directories.is_exist(path):
            if not os.listdir(path):
                return True
            else:
                return False
        else:
            logger.error('Directory does not Exist')
            return True

    @staticmethod
    def is_readable(path: str) -> bool:
        """
        | **@author:** Prathyush SP
        |
        | Check if the directory is readable
        :param path: Directory path
        :return: Bool
        """
        return os.access(path, os.R_OK)

    @staticmethod
    def is_writable(path) -> bool:
        """
        | **@author:** Prathyush SP
        |
        | Check if the directory is writable
        :param path: Directory path
        :return: Bool
        """
        return os.access(path, os.W_OK)

    @staticmethod
    def mkdir(path: str, force: bool = False):
        """
        | **@author:** Prathyush SP
        |
        | Create a directory
        :param path: Directory path
        :param force: Bool - Force Create a directory
        """
        if not Directories.is_exist(path):
            os.makedirs(path, exist_ok=True)
        elif (Directories.is_exist(path) and Directories.is_empty(path)) or force:
            Directories.rmdir(path, force=force)
            os.makedirs(path, exist_ok=True)
        else:
            logger.error('Directory already exists and is not Empty. Use force=True to force create')

    @staticmethod
    def rmdir(path: str, force: bool = False):
        """
        | **@author:** Prathyush SP
        |
        | Remove a Directory
        :param path: Directory path
        :param force: Bool - Force Remove a directory
        """
        if not Directories.is_exist(path):
            logger.error('Directory does not exist')
        elif Directories.is_exist(path) and force:
            shutil.rmtree(path)
        elif Directories.is_empty(path):
            shutil.rmtree(path)
        else:
            logger.error('Directory is not Empty. Use force=True to force remove')

    @staticmethod
    def size(path: str) -> str:
        """
        | **@author:** Prathyush SP
        |
        | Get the Size of the Directory
        :param path: Path
        :return: Size in str
        """

        if Directories.is_exist(path):
            return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')
        else:
            logger.warning('Path does not exist')
            raise FileExistsError()

    @staticmethod
    def count_dirs(path: str, depth: int = None) -> List:
        """
        | **@author:** Prathyush SP
        |
        | Count Folders in the Directory
        :param path: Path
        :param depth: Depth
        :return: List of Directory paths
        """
        if Directories.is_exist(path):
            if depth:
                return str(subprocess.check_output(["find", path, "-type", "d", "-maxdepth", '1'])).split('\\n')[1:-1]
            else:
                return str(subprocess.check_output(["find", path, "-type", "d"])).split('\\n')[1:-1]
        else:
            logger.warning('Path does not exist')
            raise FileExistsError()


class File:
    """
    | **@author:** Prathyush SP
    |
    | File Utilities
    """

    @staticmethod
    def is_exist(path: str) -> bool:
        """
        | **@author:** Prathyush SP
        |
        | Check if a file exists
        :param path: File Path
        :return: Boolean
        """
        return os.path.exists(path)

    @staticmethod
    def is_writable(path) -> bool:
        """
        | **@author:** Prathyush SP
        |
        | Check if the directory is writable
        :param path: Directory path
        :return: Bool
        """
        return os.access(path, os.W_OK)

    @staticmethod
    def is_readable(path: str) -> bool:
        """
        | **@author:** Prathyush SP
        |
        | Check if the directory is readable
        :param path: Directory path
        :return: Bool
        """
        return os.access(path, os.R_OK)

    @staticmethod
    def word_count(path: str, encoding: str = None, only_count=True) -> Union[int, Counter]:
        """
        | **@author:** Prathyush SP
        |
        | Count the number of words in a File
        :param path: File Path
        :param encoding: File Encoding
        :param only_count: True/False [Count / {Word: Count}]
        :return: Union[int, Counter]
        """
        if File.is_exist(path=path):
            if only_count:
                with open(r'' + path, 'r', encoding=encoding) as f:
                    sm = sum(v for k, v in Counter(f.read().split()).items())
                return sm
            else:
                with open(r'' + path, 'r', encoding=encoding) as f:
                    sm = Counter(f.read().split())
                return sm
        else:
            logger.error('File does not exist')
            return -1

    @staticmethod
    def line_count(path: str, encoding: str = None) -> int:
        """
        | **@author:** Prathyush SP
        |
        | Count the number of lines in a File

        :param path: File Path
        :param encoding: File Encoding
        :return: Line Count in int
        """
        if File.is_exist(path=path):
            with open(r'' + path, encoding=encoding) as f:
                ln = len(f.read().splitlines())
            return ln
        else:
            logger.error('File does not exist')
            return -1

    @staticmethod
    def read_json(path: str, encoding: str = None, object_pairs_hook: Union[dict, OrderedDict] = dict) -> Union[
        int, dict]:
        """
        | **@author:** Prathyush SP
        |
        | Load JSON (dict) from a file
        :param path: File Path
        :param encoding: File Encoding
        :return: Dictionary
        """
        if File.is_exist(path):
            try:
                with open(path, encoding=encoding) as f:
                    jl = json.load(f, object_pairs_hook=object_pairs_hook)
                return jl
            except Exception as e:
                raise Exception('Unable to load the JSON File')
        else:
            logger.error('File does not exist')
            return -1

    @staticmethod
    def size(path: str) -> int:
        """
        | **@author:** Prathyush SP
        |
        | Get File Size
        :param path: File Path
        :return: File Size
        """
        if File.is_exist(path=path):
            return os.path.getsize(path)
        else:
            logger.error('File does not exist')

    @staticmethod
    def load_json(load_path: str, file_name: str, encoding: str = None, mode: str = 'r', load_format=dict):
        """
        | **@author:** Prathyush SP
        |
        | Added Load JSON method
        :param load_path: Load Path
        :param file_name: File Name
        :param encoding: Encoding
        :param mode: Mode - 'r / rb'
        :param load_format: - Object Pair Hooks for JSON
        :return: load_format object
        """
        try:
            with open(load_path + file_name, mode=mode, encoding=encoding) as f:
                data = json.load(f, object_pairs_hook=load_format)
        except FileNotFoundError or FileExistsError:
            raise FileNotFoundError()
        return data

    @staticmethod
    def write_json(data: Union[OrderedDict, Dict], save_path: str, file_name: str, encoding: str = None, sort=False,
                   reverse=False, indent=2, parent_key: Union[str, None] = None):
        """
        | **@author:** Prathyush SP
        |
        | Write JSON to a File
        :param data: JSON Data
        :param save_path: Save Path
        :param file_name: File Name
        :param encoding: File Encoding
        :param sort: Sort JSON based on key in ascending order
        :param reverse: Sort JSON based on key in descending order
        :param indent: JSON Indentation
        :param parent_key: Parent Key
        """
        data = sorted(data.items(), key=lambda x: int(x[0]), reverse=reverse) if sort or reverse else data
        if parent_key:
            with open(r'' + save_path + '/' + file_name, 'w', encoding=encoding) as f:
                json.dump(OrderedDict([(parent_key, data)]), f, indent=indent)
        else:
            with open(r'' + save_path + '/' + file_name, 'w', encoding=encoding) as f:
                json.dump(OrderedDict(data), f, indent=indent)


class TimeUtils(object):
    """
    | **@author:** Prathyush SP
    |
    | Time Utils
    """

    class DateTime(object):
        """
        | **@author:** Prathyush SP
        |
        | Time Utils for DateTime objects
        """

        @staticmethod
        def calculate_seconds(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Seconds
            :param dt: Datetime Object
            :return: seconds
            """
            return float(
                dt.year * 12 * 30 * 24 * 60 * 60 + dt.month * 30 * 24 * 60 * 60 + dt.day * 24 * 60 * 60 + dt.hour * 60 * 60 + dt.minute * 60 + dt.second + dt.microsecond / 10 ** 6)

        @staticmethod
        def calculate_minutes(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Minutes
            :param dt: Datetime Object
            :return: Minutes
            """
            return float(TimeUtils.DateTime.calculate_seconds(dt) / 60)

        @staticmethod
        def calculate_hours(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Hours
            :param dt: Datetime Object
            :return: Hours
            """
            return float(TimeUtils.DateTime.calculate_seconds(dt) / (60 * 60))

        @staticmethod
        def calculate_days(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Days
            :param dt: Datetime Object
            :return: Days
            """
            return float(TimeUtils.DateTime.calculate_seconds(dt) / (24 * 60 * 60))

        @staticmethod
        def calculate_months(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Months
            :param dt: Datetime Object
            :return: Months
            """
            return float(TimeUtils.DateTime.calculate_seconds(dt) / (30 * 24 * 60 * 60))

        @staticmethod
        def calculate_years(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Years
            :param dt: Datetime Object
            :return: Years
            """
            return float(TimeUtils.DateTime.calculate_seconds(dt) / (12 * 24 * 60 * 60))

    class TimeDelta(object):
        """
        | **@author:** Prathyush SP
        |
        | Time Utils for TimeDelta objects
        """

        @staticmethod
        def calculate_seconds(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Seconds
            :param dt: TimeDelta Object
            :return: Seconds
            """
            return float(dt.hour * 60 * 60 + dt.minute * 60 + dt.second + dt.microsecond / 10 ** 6)

        @staticmethod
        def calculate_minutes(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Minutes
            :param dt: TimeDelta Object
            :return: Minutes
            """
            return float(TimeUtils.TimeDelta.calculate_seconds(dt) / 60)

        @staticmethod
        def calculate_hours(dt: datetime):
            """
            | **@author:** Prathyush SP
            |
            | Calculate Hours
            :param dt: TimeDelta Object
            :return: Hours
            """
            return float(TimeUtils.TimeDelta.calculate_seconds(dt) / (60 * 60))



def add_logs_to_tmp(path: str):
    """
    | **@author:** Prathyush SP
    | Add logs to the path
    :param path: Path where logs has to store
    """
    # noinspection PyProtectedMember
    with open(MODULE_CONFIG.BaseConfig._GLOBAL_LOGGING_CONFIG_FILE_PATH, 'rt') as f:
        config = yaml.safe_load(f.read())
        keys = [key for key in config['handlers'].keys()]
        for key in keys:
            if 'filename' in config['handlers'][key].keys():
                log_path = config['handlers'][key]['filename']
                log_path = log_path.split('/')
                try:
                    with open(path + '/' + log_path[-1], 'w') as log_file_path:
                        with open(config['handlers'][key]['filename']) as file_path:
                            [log_file_path.write(line) for line in file_path.readlines()]
                            os.remove(config['handlers'][key]['filename'])
                except FileNotFoundError or Exception as e:
                    # logging.warning('Logging Files not found!!')
                    pass
                config['handlers'][key]['filename'] = path + '/' + log_path[-1]
        logging.config.dictConfig(config)


def generate_uuid(name: str = '') -> str:
    """
    | **@author:** Prathyush SP
    |
    | Generate Unique ID
    :param name: UID Name
    :return: Unique ID
    """
    if name!="":
        return '_'.join([name, str(uuid.uuid4().hex)])
    else:
        return str(uuid.uuid4().hex)