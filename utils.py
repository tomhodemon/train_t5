import itertools
import numpy as np
from easydict import EasyDict
import yaml
import logging
from datetime import datetime


from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5LayerFF


def share_feed_forward(model: T5ForConditionalGeneration, 
                       share_encoder_ffn: bool = False, 
                       share_decoder_ffn: bool = False) -> T5ForConditionalGeneration:
    """
    Modify a T5 model to share feed-forward layers in encoder and/or decoder.
    
    Args:
        model: The T5 model to modify
        share_encoder_ffn: Whether to share feed-forward layers in the encoder
        share_decoder_ffn: Whether to share feed-forward layers in the decoder
    """
    if share_encoder_ffn:
        # Create a shared feed-forward layer for encoder
        shared_encoder_ff = T5LayerFF(model.encoder.config)
        # Replace all feed-forward layers in encoder blocks with the shared one
        for block in model.encoder.block:
            block.layer[1] = shared_encoder_ff

    if share_decoder_ffn:
        # Create a shared feed-forward layer for decoder
        shared_decoder_ff = T5LayerFF(model.decoder.config)
        # Replace all feed-forward layers in decoder blocks with the shared one
        for block in model.decoder.block:
            block.layer[2] = shared_decoder_ff

    return model


def scheduler_factor(k: int) -> float:
    n = 10000
    if k < n:
        return pow(n, -0.5)
    else:
        return pow(k, -0.5)
    

def generate_all_combinations(target_sum: int, combinaison_length: int, mean: int) -> np.ndarray[np.ndarray[int]]:
    values = range(1, target_sum)
    combinaisons = list(itertools.combinations_with_replacement(values, combinaison_length))

    valid_combinations = [combinaison for combinaison in combinaisons if sum(combinaison) == target_sum]

    if mean is not None:
        valid_combinations = [combinaison for combinaison in valid_combinations if np.mean(combinaison) == mean]

    return np.array(valid_combinations)


def load_config(cfg_path: str) -> EasyDict:
    with open(cfg_path, "r") as file:
        cfg = EasyDict(yaml.safe_load(file))
    return cfg


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


def get_time() -> str:
    return datetime.now().strftime("%b%d_%H-%M-%S")
