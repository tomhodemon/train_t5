import itertools
import torch.nn as nn
import numpy as np
from easydict import EasyDict
import yaml
import logging
from datetime import datetime


from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5LayerFF


def get_model(cfg: EasyDict, 
              vocab_size: int) -> T5ForConditionalGeneration:

    model_config = T5Config.from_pretrained(cfg.model.name, vocab_size=vocab_size)
    model = T5ForConditionalGeneration(model_config)
    
    if cfg.model.drop_decoder_ffn and cfg.model.share_decoder_ffn:
        print(f"Imcompatible nomenclature: {cfg.model.drop_decoder_ffn=} and {cfg.model.share_encoder_ffn=}")
        exit()

    if cfg.model.drop_encoder_ffn and cfg.model.share_encoder_ffn:
        print(f"Imcompatible nomenclature: {cfg.model.drop_encoder_ffn=} and {cfg.model.share_encoder_ffn=}")
        exit()
    
    model = share_ffn(model, cfg.model.share_encoder_ffn, cfg.model.share_decoder_ffn)
    model = drop_ffn(model, cfg.model.drop_encoder_ffn, cfg.model.drop_decoder_ffn) 

    return model

def drop_ffn(model: T5ForConditionalGeneration, 
             drop_encoder_ffn: bool = False, 
             drop_decoder_ffn: bool = False) -> T5ForConditionalGeneration:
    if drop_encoder_ffn:
        for block in model.encoder.block:
            block.layer[1] = nn.Identity()

    if drop_decoder_ffn:
        for block in model.decoder.block:
            block.layer[2] = nn.Identity()

    return model

def share_ffn(model: T5ForConditionalGeneration, 
                       share_encoder_ffn: bool = False, 
                       share_decoder_ffn: bool = False) -> T5ForConditionalGeneration:
    if share_encoder_ffn:
        shared_encoder_ff = T5LayerFF(model.encoder.config)
        for block in model.encoder.block:
            block.layer[1] = shared_encoder_ff

    if share_decoder_ffn:
        shared_decoder_ff = T5LayerFF(model.decoder.config)
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
