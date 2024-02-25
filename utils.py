import itertools
import numpy as np
from easydict import EasyDict
import yaml
import logging
from datetime import datetime

def scheduler_factor(k: int) -> float:
    n = 10000
    if k < n:
        return pow(n, -0.5)
    else:
        return pow(k, -0.5)
    
def generate_all_combinations(target_sum, combinaison_length, mean):
    values = range(1, target_sum)
    combinaisons = list(itertools.combinations_with_replacement(values, combinaison_length))

    valid_combinations = [combinaison for combinaison in combinaisons if sum(combinaison) == target_sum]

    if mean is not None:
        valid_combinations = [combinaison for combinaison in valid_combinations if np.mean(combinaison) == mean]

    return np.array(valid_combinations)


def load_config(cfg_path):
    with open(cfg_path, "r") as file:
        cfg = EasyDict(yaml.safe_load(file))
    return cfg

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def get_time():
    return datetime.now().strftime("%b%d_%H-%M-%S")