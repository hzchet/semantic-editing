from typing import Dict
import yaml

import torch
import numpy as np


def load_yaml(filename: str) -> Dict:
    """
    Loads the yaml file and returns a dict of it
    :param filename: path to file to load
    :return: dict of parsed yaml file
    """
    with open(filename, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
