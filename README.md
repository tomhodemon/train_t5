# T5 Model Training

This repository contains code for training a T5 model on `wikitext-103-v1`.

## Usage

The main training script is `train_t5.py`. You can run it with different configurations and dataset paths.

### Basic Usage

Prepare data:
```bash
python data.py
```

This will run the training with default settings:

```bash
python train_t5.py
```


### Custom Configuration

You can specify custom paths for the configuration and datasets:

```bash
python train_t5.py \
    --cfg path/to/config.yaml \
    --train_path path/to/train_dataset.pt \
    --validation_path path/to/validation_dataset.pt
```

### Command Line Arguments

- `--cfg`: Path to the configuration YAML file (default: `config/default.yaml`)
- `--train_path`: Path to the training dataset (default: `data/processed/processed_train_dataset.pt`)
- `--validation_path`: Path to the validation dataset (default: `data/processed/processed_validation_dataset.pt`)