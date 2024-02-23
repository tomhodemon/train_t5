import torch
from transformers import (T5Tokenizer,
                          T5ForConditionalGeneration,
                          T5Config,
                          Adafactor)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm

from utils import scheduler_factor, load_config, get_logger
from data import ProcessedDataset

def main(args):

    logger = get_logger('Main')
    writer = SummaryWriter()
    
    cfg = load_config(args.cfg)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} device")
    
    tokenizer = T5Tokenizer.from_pretrained(cfg.tokenizer.name)
    logger.info(f"Vocab size: {len(tokenizer)}")

    model_config = T5Config.from_pretrained(cfg.model.name, vocab_size=len(tokenizer))
    model = T5ForConditionalGeneration(model_config).to(device)
    logger.info(f"{sum(p.numel() for p in model.parameters())/1e6}M parameters")

    train_dataset = torch.load(args.train_path)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, collate_fn=ProcessedDataset.collate_fn)

    validation_dataset = torch.load(args.validation_path)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.train.batch_size, collate_fn=ProcessedDataset.collate_fn)

    # The paper does not mention any particular learning rate value. It only states:
    # "During pre-training, we use an “inverse square root” learning rate schedule... 
    # This sets a constant learning rate of 0.01 for the first 10e4 steps, 
    # then exponentially  decays the learning rate until pre-training is over."

    optimizer = Adafactor(model.parameters(), lr=cfg.optimizer.base_lr, relative_step=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: scheduler_factor(step))

    # ensures that the number of steps exceeds 50k even if we stop training at 50k

    global_step = 0

    print(next(iter(train_dataloader)))
    
    exit(0)

    for epoch in range(cfg.train.epochs):
        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            global_step += 1

            batch = {key: val.to(device) for key, val in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if global_step % cfg.train.logging_steps == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Learning Rate', scheduler.get_last_lr(), global_step)
                logger.info(f"Loss/train: {loss.item()}")

            if global_step % cfg.train.eval_steps == 0:
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    for batch in validation_dataloader:
                        batch = {key: val.to(device) for key, val in batch.items()}

                        loss = model(**batch).loss

                        total_loss+=loss.item()

                total_loss /= len(validation_dataloader)
                writer.add_scalar('Loss/Validation', total_loss, global_step)
                logger.info(f"Loss/Validation: {total_loss}")
                model.train()

            if global_step >= cfg.train.max_steps:
                break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./config/default.yaml")
    parser.add_argument("--train_path", type=str, default="data/processed/processed_train_dataset.pt")
    parser.add_argument("--validation_path", type=str, default="data/processed/processed_validation_dataset.pt")
    args = parser.parse_args()

    main(args)