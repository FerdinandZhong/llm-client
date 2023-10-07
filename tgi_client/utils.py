import logging
import sys

import yaml

version = sys.version_info
above_36 = version.major >= 3 and version.minor > 6


class ColorfulFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    green = "\x1b[32m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    formatter = (
        "%(asctime)s - {color}%(levelname)s{reset} - %(filename)s:%(lineno)d"
        " - %(module)s.%(funcName)s - %(process)d - %(message)s"
    )
    FORMATS = {
        logging.DEBUG: formatter.format(color=grey, reset=reset),
        logging.INFO: formatter.format(color=green, reset=reset),
        logging.WARNING: formatter.format(color=yellow, reset=reset),
        logging.ERROR: formatter.format(color=red, reset=reset),
        logging.CRITICAL: formatter.format(color=bold_red, reset=reset),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def register_logger(logger=None):
    """register colorful debug log"""
    if not logger:
        logger = logging.getLogger()
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(ColorfulFormatter())
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)


def read_yaml(yaml_file_path):
    """Reads a Yaml file"""
    with open(yaml_file_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
