from curious_agent.util.custom_json_encoder import CustomJsonEncoder
import json
from munch import Munch


def pipeline_config_loader(config_file_path):
    data = json.load(open(config_file_path))
    for k,v in data.items():
        if isinstance(v, dict):
            data[k] = Munch(v)
    return Munch(data)
