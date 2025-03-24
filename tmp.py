from transformers import (T5Tokenizer,
                          T5ForConditionalGeneration,
                          T5Config)

import datasets

from utils import load_config

if __name__ == "__main__":
    cfg = load_config("config/dev.yaml")

    tokenizer = T5Tokenizer.from_pretrained(cfg.tokenizer.name)

    model_config = T5Config.from_pretrained(cfg.model.name, vocab_size=len(tokenizer))
    model = T5ForConditionalGeneration(model_config)

    print(model.config)
    print(model)

    ds = datasets.load_dataset('wikitext', 'wikitext-103-v1')
    print(ds)
