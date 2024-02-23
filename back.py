class ProcessedDataset:
    def __init__(self,
                 dataset,
                 tokenizer,
                 cfg):
        super(ProcessedDataset).__init__()
        
        self.tokenizer = tokenizer
        self.dataset = dataset

        self.input_length = cfg.data.input_length

        target_sum = int(round(self.input_length*cfg.data.corruption_rate)) # if max_seq_len = 128, round(141*0.15) = 21
        combinaison_length = int(round(target_sum/cfg.data.mean_corrupted_span_length)) # if max_seq_len = 128, 21/3 = 7
        self.corrupted_span_combinaisons = generate_all_combinations(target_sum, combinaison_length, cfg.data.mean_corrupted_span_length)
        seq_1_length = combinaison_length//2
        seq_1_target_sum = (self.input_length - target_sum)//2
        self.token_span_combinaisons_1 = generate_all_combinations(seq_1_target_sum, seq_1_length, None)

        seq_2_length = combinaison_length - seq_1_length
        seq_2_target_sum = (self.input_length - target_sum) - seq_1_target_sum
        self.token_span_combinaisons_2 = generate_all_combinations(seq_2_target_sum, seq_2_length, None)

        self.sentinel_ids = [self.tokenizer.encode(special_token)[0] for special_token in tokenizer.additional_special_tokens[:combinaison_length]]

        # pre-processing
        # only retaine lines that contained at least 5 words (from paper)
        self.reducer = lambda example: len(example['text'].split(' ')) >= 5
        self.dataset = dataset.filter(self.reducer)

        tokenized_dataset = dataset.map(
            self._tokenize,
            batched=True,
            batch_size=1000,
            drop_last_batch=False,
            remove_columns=['text']
        )

        packed_and_tokenized_dataset =  tokenized_dataset.map(
            self._pack,
            batched=True,
            batch_size=1000,
            drop_last_batch=False,
        )

        self.processed_dataset = packed_and_tokenized_dataset.map(
            self._mask,
            batched=False,
            drop_last_batch=False,
        )

    def _tokenize(self, batch):
        return self.tokenizer(
            batch['text'],
            return_attention_mask=False,
        )

    def _pack(self, batch):
        input_ids = np.concatenate(batch['input_ids'])
        len_input_ids = len(input_ids)
        input_ids = input_ids[:-(len_input_ids%self.input_length)]
        input_ids = input_ids.reshape((-1, self.input_length))
        return {'input_ids': input_ids}

    def _mask(self, sample):
        input_ids = sample['input_ids']
        corrupted_seq = self._get_random_span_seq(self.corrupted_span_combinaisons)
        token_seq = np.concatenate((
                self._get_random_span_seq(self.token_span_combinaisons_1),
                self._get_random_span_seq(self.token_span_combinaisons_2)))

        corrupted_input_ids = []
        label_input_ids = []

        for i, (c, t) in enumerate(zip(corrupted_seq, token_seq)):
            corrupted_input_ids.extend(input_ids[:t])

            corrupted_input_ids.append(self.sentinel_ids[i]) # sentienl_ids
            input_ids = input_ids[t:]

            label_input_ids.append(self.sentinel_ids[i])
            label_input_ids.extend(input_ids[:c])

            input_ids = input_ids[c:]

        corrupted_input_ids.append(1) # <EOS> token
        label_input_ids.append(1) # <EOS> token

        return {
            'input_ids': corrupted_input_ids,
            'labels': label_input_ids }

    def _get_random_span_seq(self, combinaison):
        idx = random.randint(0, len(combinaison)-1)
        random_span_seq = combinaison[idx]
        np.random.shuffle(random_span_seq)
        return random_span_seq
    
    def get_processed_dataset(self) -> datasets.Dataset:
        return self.processed_dataset