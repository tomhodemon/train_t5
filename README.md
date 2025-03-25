# T5 Model Training

This repository contains code for training a T5 model on `wikitext-103-v1`.

## Usage

The main training script is `train_t5.py`. You can run it with different configurations and dataset paths.

### Basic Usage

Prepare data:
```bash
python data.py
```

This will start the training:

```bash
python train_t5.py \
    --cfg path/to/config.yaml \
    --train_path path/to/train_dataset.pt \
    --validation_path path/to/validation_dataset.pt
```

## Future Work
- [ ] Mutli GPU training support

