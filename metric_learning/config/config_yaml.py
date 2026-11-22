import os
import yaml
import argparse

"""
transform yaml file to config object
"""


def _expand_metric_data_root(value):
    if not isinstance(value, str) or "${METRIC_DATA_ROOT}" not in value:
        return value
    if "METRIC_DATA_ROOT" not in os.environ:
        raise RuntimeError("METRIC_DATA_ROOT is not set")
    return value.replace("${METRIC_DATA_ROOT}", os.environ["METRIC_DATA_ROOT"])


class Config:
    def __init__(self, **response):
        for k, v in response.items():
            if isinstance(v, dict):
                self.__dict__[k] = Config(**v)
            else:
                self.__dict__[k] = _expand_metric_data_root(v)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config file")
    args = parser.parse_args()

    config_file = args.config

    with open(config_file, 'r') as f:
        config_node = yaml.safe_load(f)
    config = Config(**config_node)
    print(config)
