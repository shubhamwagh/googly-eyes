import os
import yaml

CONFIG_DIR = "config"


def load_config(config_dir=CONFIG_DIR):
    """
    Load configuration settings from a YAML file.

    Parameters:
    - config_dir (str): The directory path where the configuration file is located.
                       Default is the 'config' directory in the current working directory.

    Returns:
    - dict: A dictionary containing the configuration settings loaded from the YAML file.
    """

    config_filename = "config.yml"
    yaml_path = os.path.join(config_dir, config_filename)
    with open(yaml_path, 'r') as stream:
        return yaml.safe_load(stream)
