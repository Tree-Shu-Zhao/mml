# MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition

This repository contains the official implementation of the paper "MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition".

## Installation
We suggest use `uv` to manage environments.
```bash
# Clone the repository
git clone [repository-url]
cd MoRA_paper

uv sync
source .venv/bin/activate
```

## Data Preparation

Please refer to [DATA.md](DATA.md) for detailed instructions on how to organize the datasets. After downloading datasets, using `bash scripts/preprocess.sh` to preprocess datasets.

## Usage

### Training

```bash
# Train on Hateful Memes dataset. Other datasets can refer to this script
bash scripts/hatememes.sh
```

### Testing

```bash
# Test a trained model
python src/main.py experiment=<your_experiment_config> test.TEST_ONLY=True test.CHECKPOINT_PATH=/path/to/checkpoint
```