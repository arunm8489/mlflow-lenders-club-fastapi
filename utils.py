import yaml


def read_config(config_path):
    with open(config_path) as cfg_file:
        content = yaml.safe_load(cfg_file)
    return content


