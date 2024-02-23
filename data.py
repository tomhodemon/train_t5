import torch
import numpy as np
import datasets
from transformers import T5Tokenizer
from utils import generate_all_combinations, load_config, get_logger

import os
from typing import Optional
from easydict import EasyDict
import random

class Builder:
    def __init__(self, 
                 cfg: EasyDict, 
                 tokenizer: T5Tokenizer) -> None:
        
        self.cfg = cfg
        self.tokenizer = tokenizer

        self._dataset = None
        self._processed_dataset = None
        self.split = None

        self.logger = get_logger('Builder')

    def load_dataset(self, split: str) -> None:
        self.logger.info(f'Loading dataset with split: {split}')
        self._dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1', split=split)
        self.split = str(self._dataset.split)
        self.logger.info('Dataset loaded successfully')
    
    def process_dataset(self) -> None:
        if self._dataset is None:
            self.logger.error('You need to load the dataset first')
            raise ValueError('You need to load the dataset first')
        
        self.logger.info('Processing dataset')
        
        def _tokenize(batch):
            return self.tokenizer(
                batch['text'],
                return_attention_mask=False,
            )

        def _pack(batch, input_length):
            input_ids = np.concatenate(batch['input_ids'])
            len_input_ids = len(input_ids)
            input_ids = input_ids[:-(len_input_ids%input_length)]
            input_ids = input_ids.reshape((-1, input_length))
            return {'input_ids': input_ids}
        
        def _get_random_span_seq(combinaison):
            idx = random.randint(0, len(combinaison)-1)
            random_span_seq = combinaison[idx]
            np.random.shuffle(random_span_seq)
            return random_span_seq
        
        def _mask(sample, corrupted_span_combinaisons, token_span_combinaisons_1, token_span_combinaisons_2, sentinel_ids):
            input_ids = sample['input_ids']
            corrupted_seq = _get_random_span_seq(corrupted_span_combinaisons)
            token_seq = np.concatenate((
                    _get_random_span_seq(token_span_combinaisons_1),
                    _get_random_span_seq(token_span_combinaisons_2)))

            corrupted_input_ids = []
            label_input_ids = []

            for i, (c, t) in enumerate(zip(corrupted_seq, token_seq)):
                corrupted_input_ids.extend(input_ids[:t])

                corrupted_input_ids.append(sentinel_ids[i]) # sentienl_ids
                input_ids = input_ids[t:]

                label_input_ids.append(sentinel_ids[i])
                label_input_ids.extend(input_ids[:c])

                input_ids = input_ids[c:]

            corrupted_input_ids.append(1) # <EOS> token
            label_input_ids.append(1) # <EOS> token

            return {
                'input_ids': corrupted_input_ids,
                'labels': label_input_ids }

        input_length = self.cfg.data.input_length

        target_sum = int(round(input_length*self.cfg.data.corruption_rate)) # if max_seq_len = 128, round(141*0.15) = 21
        combinaison_length = int(round(target_sum/self.cfg.data.mean_corrupted_span_length)) # if max_seq_len = 128, 21/3 = 7
        corrupted_span_combinaisons = generate_all_combinations(target_sum, combinaison_length, self.cfg.data.mean_corrupted_span_length)
        seq_1_length = combinaison_length//2
        seq_1_target_sum = (input_length - target_sum)//2
        token_span_combinaisons_1 = generate_all_combinations(seq_1_target_sum, seq_1_length, None)

        seq_2_length = combinaison_length - seq_1_length
        seq_2_target_sum = (input_length - target_sum) - seq_1_target_sum
        token_span_combinaisons_2 = generate_all_combinations(seq_2_target_sum, seq_2_length, None)

        sentinel_ids = [self.tokenizer.encode(special_token)[0] for special_token in tokenizer.additional_special_tokens[:combinaison_length]]

        # pre-processing
        # only retaine lines that contained at least 5 words (from paper)
        reducer = lambda example: len(example['text'].split(' ')) >= 5
        self._dataset = self._dataset.filter(reducer)

        tokenized_dataset = self._dataset.map(
            _tokenize,
            batched=True,
            batch_size=1000,
            drop_last_batch=False,
            remove_columns=['text']
        )

        packed_and_tokenized_dataset = tokenized_dataset.map(
            _pack,
            batched=True,
            batch_size=1000,
            drop_last_batch=False,
            fn_kwargs={'input_length': input_length}
        )

        self._processed_dataset = packed_and_tokenized_dataset.map(
            _mask,
            batched=False,
            drop_last_batch=False,
            fn_kwargs={
                'corrupted_span_combinaisons': corrupted_span_combinaisons,
                'token_span_combinaisons_1': token_span_combinaisons_1,
                'token_span_combinaisons_2': token_span_combinaisons_2,
                'sentinel_ids': sentinel_ids
            }
        )

        self._processed_dataset = ProcessedDataset(self._processed_dataset, self.cfg)
        self.logger.info('Dataset processed successfully')

    def save_dataset(self, path: Optional[str] = None, name: Optional[str] = None) -> None:
        if self._processed_dataset is None:
            self.logger.error('You need to process the dataset first')
            raise ValueError('You need to process the dataset first')
        
        if path is None:
            path = self.cfg.data.save_path

        if name is None:
            name = f'processed_{self.split}_dataset.pt'

        self.logger.info(f'Saving dataset to path: {path} with name: {name}')
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self._processed_dataset, os.path.join(path, name))
        self.logger.info('Dataset saved successfully')

    def get_processed_dataset(self) -> datasets.Dataset:
        if self._processed_dataset is not None:
            return self._processed_dataset
        else:
            self.logger.error('You need to process the dataset first')
            raise ValueError('You need to process the dataset first')
        
    def __repr__(self) -> str:
        tokenizer_info = f'Tokenizer={self.tokenizer.__class__.__name__}'
        return f'Builder(cfg={self.cfg.name}, {tokenizer_info}, {repr(self._processed_dataset)})'
        
class ProcessedDataset(datasets.Dataset):
    def __init__(self, 
                 dataset: datasets.Dataset,
                 cfg: EasyDict):
        super(ProcessedDataset).__init__()
        self.dataset = dataset
        self.cfg = cfg

    @staticmethod   
    def collate_fn(batch):
        print(batch)
        Xs = [torch.LongTensor(item['input_ids']) for item in batch]
        ys = [torch.LongTensor(item['labels']) for item in batch]

        X_tensors = torch.stack(Xs)
        y_tensors = torch.stack(ys)

        return {
            'input_ids': X_tensors,
            'labels': y_tensors
        }
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]
    
    def __repr__(self) -> str:
        dataset_info_dict = {
                'path': 'wikitext',
                'name': 'wikitext-103-v1',
                'split': str(self.dataset.split)
            }
        dataset_info = f'Dataset={str(dataset_info_dict)}'
        return f'ProcessedDataset(cfg={self.cfg.name}, {dataset_info})'
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config/default.yaml')
    args = parser.parse_args()
    cfg = load_config(args.cfg)

    tokenizer = T5Tokenizer.from_pretrained(cfg.tokenizer.name)
    
    builder = Builder(cfg, tokenizer)
    
    builder.load_dataset(split='train')
    builder.process_dataset()
    builder.save_dataset()

    builder.load_dataset(split='validation')
    builder.process_dataset()
    builder.save_dataset()