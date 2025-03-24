import torch
from transformers import (T5Tokenizer,
                          T5ForConditionalGeneration,
                          T5Config,
                          Adafactor)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm

from utils import scheduler_factor, load_config, get_logger, get_time
from data import ProcessedDataset


def main(args):
    cfg = load_config(args.cfg)

    logger = get_logger('main')
    writer = SummaryWriter(f'runs/{get_time()}_T5-config-{cfg.name}', flush_secs=30)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using {device} device')
    
    tokenizer = T5Tokenizer.from_pretrained(cfg.tokenizer.name)
    logger.info(f'Vocab size: {len(tokenizer)}')

    model_config = T5Config.from_pretrained(cfg.model.name, vocab_size=len(tokenizer))
    model = T5ForConditionalGeneration(model_config).to(device)
    logger.info(f"{sum(p.numel() for p in model.parameters())/1e6}M parameters")

    train_dataset = torch.load(args.train_path, weights_only=False)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, collate_fn=ProcessedDataset.collate_fn)

    validation_dataset = torch.load(args.validation_path, weights_only=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.train.batch_size, collate_fn=ProcessedDataset.collate_fn)

    optimizer = Adafactor(model.parameters(), lr=cfg.optimizer.base_lr, relative_step=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: scheduler_factor(step))

    global_step = 0

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
                writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_step)

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
                model.train()

            if global_step >= cfg.train.max_steps:
                break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/default.yaml")
    parser.add_argument("--train_path", type=str, default="data/processed/processed_train_dataset.pt")
    parser.add_argument("--validation_path", type=str, default="data/processed/processed_validation_dataset.pt")
    args = parser.parse_args()

    main(args)
