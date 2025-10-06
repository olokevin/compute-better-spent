# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements structured matrix replacements for dense linear layers in neural networks, specifically Block Tensor-Train (BTT), Tensor-Train (TT), Monarch, and Kronecker matrices. The work demonstrates that BTT matrices can match or exceed dense layer performance with significantly less compute on CIFAR-10/100, ImageNet, and GPT-2 language modeling tasks.

## Environment Setup

Two separate conda environments are required:

### struct environment (CIFAR-10/100 and ImageNet)
```bash
source setup.sh  # Creates both environments
conda activate struct
```
Dependencies include PyTorch 1.12.0, FFCV for fast data loading, and CoLA (Compositional Linear Algebra library).

### gpt environment (GPT-2 experiments)
```bash
source setup.sh  # If not already run
conda activate gpt
```
Includes PyTorch with CUDA 11.8, transformers, and CoLA.

**Important**: The setup script clones CoLA to `$HOME/cola` and patches it for PyTorch 1.12 compatibility. Both environments use this shared CoLA installation.

## Dataset Preparation

### CIFAR-10/100 (requires struct environment)
```bash
conda activate struct
python scaling_mlps/data_utils/dataset_to_beton.py --dataset_name cifar10 --mode train --res 32
python scaling_mlps/data_utils/dataset_to_beton.py --dataset_name cifar10 --mode val --res 32
python scaling_mlps/data_utils/dataset_to_beton.py --dataset_name cifar100 --mode train --res 32
python scaling_mlps/data_utils/dataset_to_beton.py --dataset_name cifar100 --mode val --res 32
```
This creates `.beton` files in the `beton/` directory for fast data loading with FFCV.

### OpenWebText (requires gpt environment)
```bash
conda activate gpt
python prepare_owt.py
```
Downloads and tokenizes the OpenWebText dataset to `data/open/`.

## Training Commands

### CIFAR experiments (MLP)
```bash
conda activate struct
python3 train_cifar.py \
  --dataset=cifar10 \
  --model=MLP \
  --width=64 \
  --depth=3 \
  --lr=3e-3 \
  --batch_size=1024 \
  --epochs=500 \
  --struct=btt \
  --layers=all_but_last \
  --scale_factor=8 \
  --input_lr_mult=0.1 \
  --scheduler=cosine
```

### CIFAR experiments (ViT)
```bash
conda activate struct
python3 train_cifar.py \
  --dataset=cifar10 \
  --model=ViT \
  --vit_depth=6 \
  --vit_patch_size=4 \
  --lr=1e-3 \
  --batch_size=512 \
  --epochs=500 \
  --struct=btt \
  --layers=all_but_last
```

### GPT-2 training (single GPU)
```bash
conda activate gpt
python train_gpt.py config/train_open.py --struct=btt_norm --layers=all --d_model=1024 --tt_rank=4 --batch_size=12
```

### GPT-2 training (distributed, 8 GPUs)
```bash
conda activate gpt
torchrun --nproc_per_node=8 train_gpt.py config/train_open.py \
  --struct=btt_norm \
  --layers=all \
  --d_model=1024 \
  --tt_rank=4 \
  --n_layer=12 \
  --n_head=6 \
  --d_head=64 \
  --max_iters=600_000 \
  --batch_size=12 \
  --gradient_accumulation_steps=40 \
  --init_lr=2e-3
```

### Running experiment sweeps
```bash
# CIFAR scaling laws
sh experiments/mlp.sh
sh experiments/vit.sh

# ImageNet experiments
sh experiments/imagenet.sh

# GPT-2 scaling laws
sh experiments/gpt.sh

# Structure-aware learning rate experiments
sh experiments/check_dh.sh
sh experiments/lr_landscape.sh
```

## Visualization

After training, visualize results using Jupyter notebooks:
```bash
conda activate struct
jupyter notebook notebooks/scaling_laws.ipynb       # Scaling law plots
jupyter notebook notebooks/struct_aware_lr.ipynb    # Learning rate landscape figures
```

## Architecture Overview

### Core Modules

**`/nn`**: Neural network model implementations
- `cola_nn.py`: Core `CoLALayer` wrapper that converts CoLA LinearOperators to nn.Module layers with proper initialization and learning rate scaling
- `fcnet.py`: MLP implementation for CIFAR experiments
- `vit.py`: Vision Transformer implementation
- `gpt2.py`: GPT-2 model adapted from nanoGPT with structured matrix support

**`/ops`**: Structured matrix operator implementations using CoLA framework
- `operators.py`: Defines BTT, TT, Monarch, and Permutation operators as CoLA LinearOperators
- These operators are parameterized by low-rank cores/factors and have efficient forward passes

**`/mm`**: Low-level custom CUDA/PyTorch kernels for matrix operations
- `btt_mvm.py`: Optimized BTT matrix-vector multiplication
- `blockdiag_butterfly_multiply.py`: Block-diagonal butterfly multiplication for Monarch matrices

**`/model`**: Training utilities and configuration system
- `fns.py`: Training/evaluation loops, metric tracking
- `gpt_fns.py`: GPT-2 specific utilities (learning rate scheduling, distributed training setup)
- `configurator.py`: Configuration override system (see below)

**`/scaling_mlps`**: Forked utilities from scaling_mlps repo for CIFAR experiments

**`/timm`**: Forked from pytorch-image-models for ImageNet training

### Key Architectural Concepts

**Structured Matrices**: The repository implements several structured matrix families:
- **BTT (Block Tensor-Train)**: Novel family containing Monarch matrices, superior performance
- **TT (Tensor-Train)**: Classic tensor decomposition
- **Monarch**: Two block-diagonal butterfly matrices composed together
- **Kronecker**: Product of smaller matrices
- **Low-rank**: Standard rank-r factorization

All are implemented as CoLA LinearOperators in `/ops/operators.py` and wrapped via `CoLALayer` in `/nn/cola_nn.py`.

**Structure-Aware Scaling**: Different structured matrices require different initialization scales and learning rates. The code implements:
- µP (Maximal Update Parameterization) initialization in `cola_nn.py:cola_init()`
- Per-layer learning rate multipliers stored in `lr_mult` attributes on parameters
- Learning rate adjustment functions in `model/gpt_fns.py:get_lr_mult()` and `update_lrs()`

**CoLA Parameterization**: The `cola_parameterize()` function in `nn/cola_nn.py` traverses a model and replaces specified `nn.Linear` layers with `CoLALayer` instances based on the `struct` and `layers` arguments.

## Configuration System

Training scripts use a "poor man's configurator" (`model/configurator.py`) that:
1. Sets default hyperparameters as global variables in the training script
2. Executes a config file (e.g., `config/train_open.py`) to override defaults
3. Parses command-line `--key=value` arguments to override further

Example: `python train_gpt.py config/train_open.py --d_model=1024 --init_lr=2e-3`

This runs `config/train_open.py` which sets base values, then overrides `d_model` and `init_lr` from command line.

## Important Parameters

**Structure selection** (`--struct`):
- `dense`: Standard dense matrices
- `btt` or `btt_norm`: Block Tensor-Train (recommended)
- `tt`: Tensor-Train with `--tt_rank` to set rank
- `monarch`: Monarch matrices
- `kron`: Kronecker product
- `low_rank`: Low-rank factorization with `--rank_frac` to set rank fraction

**Layer replacement** (`--layers`):
- `all`: Replace all linear layers
- `all_but_last`: Replace all except final classifier/output layer
- `ffn`: Replace only feedforward network layers

**Scaling** (`--scale_factor`): Multiplier for model width/size while keeping parameter count approximately constant

**Learning rate multipliers**:
- `--input_lr_mult`: Learning rate multiplier for first layer (typically 0.1 for structured matrices)
- Layer-specific multipliers are automatically computed based on structure and µP

## Checkpoints and Logging

- Checkpoints saved to `checkpoints/` or custom `--out_dir`
- WandB logging available with `--wandb_log` and `--wandb_project` flags
- Training scripts log model FLOPS, parameter counts, and memory usage via `get_model_summary_and_flops()` in `cola_nn.py`

## Testing Individual Components

To test a specific structured matrix:
```python
from ops.operators import BTT, TT, Monarch
from nn.cola_nn import CoLALayer
import torch

# Create BTT operator
Ms = [torch.randn(1, 1, 4, 4) for _ in range(3)]  # Example cores
btt_op = BTT(Ms)

# Wrap as layer
layer = CoLALayer(btt_op)

# Forward pass
x = torch.randn(2, btt_op.shape[0])  # batch_size=2
y = layer(x)
```

## Common Workflows

**Adding a new structured matrix type**:
1. Implement as a CoLA `LinearOperator` in `ops/operators.py` with `_rmatmat()` method
2. Add factory function to `cola_nn.py:cola_parameterize()` to create instances
3. Add initialization logic to `cola_nn.py:cola_init()` with appropriate std scaling
4. Add to struct type checks in training scripts if special handling needed

**Debugging training runs**:
1. Check that FLOP count printed at start matches expectations for structure
2. Verify learning rate multipliers are applied correctly (logged at initialization)
3. For GPT-2, ensure `gradient_accumulation_steps` is a multiple of world size for DDP
4. For CIFAR, verify `.beton` files exist in correct location

**Reproducing paper results**:
1. Run the experiment scripts in `/experiments` directory
2. Scripts sweep over `scale_factor` values to generate scaling law curves
3. Use notebooks to visualize and compare structures
