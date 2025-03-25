import torch
from transformers import (T5Tokenizer,
                          T5ForConditionalGeneration,
                          T5Config,
                          Adafactor)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm

from utils import scheduler_factor, load_config, get_logger, get_time, get_model
from data import ProcessedDataset
    

def main(args):
    cfg = load_config(args.cfg)

    logger = get_logger('main')
    logger.info(f"Config: {cfg}")

    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed(cfg.train.seed)

    run_name = f'{get_time()}_T5-cfg-{cfg.name}-drop_enc_ffn-{cfg.model.drop_encoder_ffn}-drop_dec_ffn-{cfg.model.drop_decoder_ffn}-share_enc_ffn-{cfg.model.share_encoder_ffn}-share_dec_ffn-{cfg.model.share_decoder_ffn}'
    writer = SummaryWriter(f'runs/{run_name}', flush_secs=30)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using {device} device')
    
    tokenizer = T5Tokenizer.from_pretrained(cfg.tokenizer.name)
    logger.info(f'Vocab size: {len(tokenizer)}')

    model = get_model(cfg, len(tokenizer))
    model = model.to(device)

    logger.info(f"Model nomenclature: {cfg.model.drop_decoder_ffn=} {cfg.model.drop_encoder_ffn=} {cfg.model.share_encoder_ffn=} {cfg.model.share_decoder_ffn=}")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())/1e2}M parameters")

    train_dataset = torch.load(args.train_path, weights_only=False)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, collate_fn=ProcessedDataset.collate_fn, shuffle=False)

    validation_dataset = torch.load(args.validation_path, weights_only=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.train.batch_size, collate_fn=ProcessedDataset.collate_fn, shuffle=False)

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
