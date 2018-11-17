import logging.config
import yaml


def setup_logging(logging_file: str):
    with open(logging_file) as logging_config_file:
        config = yaml.safe_load(logging_config_file.read())
    logging.config.dictConfig(config)
