# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Paper Summary

**Title**: "Compute Better Spent: Replacing Dense Layers with Structured Matrices" (ICML 2024)

**Authors**: Shikai Qiu*, Andres Potapczynski*, Marc Finzi, Micah Goldblum, Andrew Gordon Wilson (NYU & CMU)

### Key Contributions

1. **Block Tensor-Train (BTT) Matrix Family**: Novel structured matrix family that contains Monarch matrices as a special case. BTT achieves better scaling laws than dense matrices across multiple tasks:
   - CIFAR-10/100: Exponentially lower training loss than dense for MLPs and ViTs
   - ImageNet-1k: Matches dense ViT-S/32 performance with 3.8× less compute
   - GPT-2: More efficient than dense for training small language models

2. **Structure-Aware Optimization**: Systematic method for determining optimal initialization and learning rates for structured matrices based on µP (Maximal Update Parameterization):
   - Different structures require drastically different learning rates (often O(√d) larger than dense)
   - Automatic scaling rules derived by decomposing structured matrices into compositions of smaller dense matrices
   - Critical for realizing performance benefits - without structure-aware scaling, benefits don't emerge

3. **Scaling Law Analysis**: Empirical demonstration that structured matrices can achieve better scaling exponents α in power law E ∝ C^(-α):
   - Challenges prior belief that scaling exponents are determined solely by task/data
   - Structures without parameter sharing (BTT, Monarch) consistently outperform those with sharing (Kronecker, TT)
   - Key principle: matching parameter count to FLOPs leads to better scaling

4. **Compute-Memory Trade-off**: Analysis of compute per dimension ξ = C/d as key hyperparameter:
   - Lower ξ (lower rank BTT, more blocks in Monarch) → better compute-efficiency
   - Higher ξ → better memory-efficiency (fewer activations to store)
   - Dense matrices are most memory-efficient but least compute-efficient

5. **Weight Normalization for Stability**: Technique to prevent unbounded activation growth in transformers with structured matrices:
   - Normalizes BTT cores to have RMS ≤ initialization scale
   - Learnable scalar multipliers allow singular values to grow if needed
   - Essential for stable training of GPT-2 models with BTT

### Main Insights

- **Structured matrices enable different compute allocations**: For same compute C, structured layers can be exponentially wider than dense (e.g., d ∝ C^(2/3) for block diagonal vs d ∝ C^(1/2) for dense)
- **Structure-dependent scaling laws**: Different matrix structures lead to different scaling exponents, with BTT showing best performance
- **Parameter sharing hurts scaling**: Structures that reuse parameters (Kronecker, TT) underperform those with parameters equal to FLOPs
- **Optimal compute per dimension**: On CIFAR, ξ = O(√d) for BTT (rank-1, 2 cores) achieves best compute-efficiency

## Overview

This repository implements the above contributions, specifically structured matrix replacements for dense linear layers in neural networks: Block Tensor-Train (BTT), Tensor-Train (TT), Monarch, Kronecker, and Low-Rank matrices. The work demonstrates that BTT matrices can match or exceed dense layer performance with significantly less compute on CIFAR-10/100, ImageNet, and GPT-2 language modeling tasks.

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

### CIFAR with Zeroth-Order (ZO) Training
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
  --ZO_config_path=ZO_Estim/ZO_config_cifar.yaml
```
Note: ZO training estimates gradients via forward-only perturbations instead of backpropagation.

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

**`/ZO_Estim`**: Zeroth-order (ZO) gradient estimation framework
- `ZO_Estim_MC.py`: Monte Carlo gradient estimator using parameter or activation perturbation
- `ZO_Estim_entry.py`: Entry point with `build_ZO_Estim()` and objective function definitions
- `ZO_utils.py`: Helper classes (`SplitedLayer`, `SplitedParam`) and forward/backward hooks
- See `ZO_Estim/README_ZO.md` for detailed documentation

### Key Architectural Concepts

**Structured Matrices**: The repository implements several structured matrix families:
- **BTT (Block Tensor-Train)**: Novel family from the paper, contains Monarch matrices as special case (rank-1, 2 cores, √d blocks). Defined by cores with no parameter sharing along block dimensions. For c=2 cores: y_αβ = Σ_γσ L_αβγσ R_σβγδ x_γδ
  - Implementation: `ops/operators.py:BTT`
  - Key property: Parameters = FLOPs = 2rd^(3/2) for rank r, enabling compute-efficient scaling
  - Expressivity: Rank r=√d can represent any dense matrix (unlike TT which needs r=d)

- **TT (Tensor-Train)**: Classic tensor decomposition with parameter sharing. For c=2 cores: y_αβ = Σ_γσ L_αγσ R_σβδ x_γδ
  - Each parameter used √d times (parameter sharing along block dimensions β, γ)
  - Parameters: 2rd, FLOPs: 2rd^(3/2), so uses each parameter √d times

- **Monarch**: Two block-diagonal butterfly matrices composed: PLP⊤R. Special case of BTT with rank-1 and √d blocks
  - Implementation: Equivalent to BTT with specific parameterization
  - Parameters = FLOPs = 2d^(3/2) for √d blocks

- **Kronecker**: Product of smaller matrices L⊗R. Strong parameter sharing assumptions (objects of same kind)
  - Parameters: 2d, FLOPs: 2d^(3/2), uses each parameter √d times

- **Low-rank**: Standard rank-r factorization W=UV
  - Dimensionality reduction assumption (subspace relevance)
  - Parameters = FLOPs = 2rd

All are implemented as CoLA LinearOperators in `/ops/operators.py` and wrapped via `CoLALayer` in `/nn/cola_nn.py`.

**Structure-Aware Scaling (Paper Section 3)**: Different structured matrices require different initialization scales and learning rates. The code implements the paper's µP-based approach:

- **Learning rate multipliers** (Table 2 in paper):
  - Dense: η ∝ 1/d
  - Low-rank UV: κ_U = d/2r, κ_V = 1/2
  - Kronecker L⊗R: κ_L = √d/2, κ_R = √d/2
  - Monarch PLP⊤R: κ_L = b/2, κ_R = b/2 (b = number of blocks)
  - TT(L,R): κ_L = √d/2r, κ_R = √d/2
  - BTT(L,R): κ_L = √d/2r, κ_R = √d/2

- **Implementation locations**:
  - `cola_nn.py:cola_init()`: µP initialization with spectral norm scaling
  - `cola_nn.py:cola_parameterize()`: Assigns learning rate multipliers κ to parameters
  - `model/gpt_fns.py:get_lr_mult()` and `update_lrs()`: Apply multipliers during training
  - Per-layer learning rate multipliers stored in `lr_mult` attributes on parameters

- **Key insight**: Decompose structured matrix multiply as Wx = G_k P_k ... G_1 P_1 x where G_i are dense components and P_i are fixed norm-preserving transforms. Apply µP to each G_i independently based on its dimensions.

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

**Paper-to-Code Mapping**:
- **Figure 1b, Figure 4 (Scaling Laws)**: Generated by `notebooks/scaling_laws.ipynb` from runs logged to WandB
- **Figure 2 (Structure-aware LR)**: Generated by scripts in `experiments/check_dh.sh` and `experiments/lr_landscape.sh`
- **Figure 5 (Compute per dimension)**: Varies `--tt_rank` for BTT or number of blocks for Monarch
- **Figure 7 (Weight normalization)**: Weight norm implemented in `nn/cola_nn.py:CoLALayer` for GPT-2 stability
- **Figure 8 (ImageNet ViT)**: Generated by `experiments/imagenet.sh`
- **Figure 9 (GPT-2)**: Generated by `experiments/gpt.sh` with `--struct=btt_norm`
- **Section 3.2 (µP for structures)**: Core algorithm in `nn/cola_nn.py:cola_parameterize()` and `cola_init()`
- **Section 5.1 (Weight normalization)**: RMS normalization with learnable scalars in `nn/cola_nn.py`
- **Appendix C (BTT expressivity)**: Projection algorithm not explicitly implemented but proven theoretically
- **Appendix E (Hyperparameters)**: Default values set in `scaling_mlps/utils/parsers.py` and config files

**Using Zeroth-Order (ZO) Training**:
1. Create a config file (see `ZO_Estim/ZO_config.yaml` for reference)
2. Implement task-specific objective function in `ZO_Estim/ZO_Estim_entry.py:build_obj_fn()`
3. Pass `--ZO_config_path=path/to/config.yaml` to training script
4. ZO training replaces backpropagation with gradient estimation via perturbations
5. See `ZO_Estim/README_ZO.md` for full documentation and workflow


## Zeroth-order (ZO) gradient estimator
implement under directory ZO_grad_estimator

ZO use repeat forward (loss computation) to estimate the gradients: one "clean" forward, and one (or multiple) "perturbed" forwards. It replaces the forward and backward, directly assign the gradient estimation to param.grad, and later we can still call the optimizer.step() to update the parameters

* config and init 
  * use a separate .yaml file to save the configs for 
  * in configs, use rules/ to specify the name_pattern, which help to find the param / layer we want to estimate the gradients when 

* obj_fn
  * we create the form of obj_fn for different tasks
  * and update the new data/labels into the obj_fn in each iteration

* gradient estimation
  * we have two forms of gradient estimation: weight perturbation and node perturbation
  * weight perturbation
    * generate random perturbation of the same size as the params
    * add to these params during perturbed forwards
    * need to be in memory-efficient implementation: do not explicily save the perturbation. instead, just save the seen when generating the perturbation, and use the same seed to re-generate it when assigning the gradient estimation to param.grad
  * node perturbation
    * generate random perturbation of the same size as the output activation
    * add to the output activation during perturbed forwards
    * always don't save the activations, though it is needed when computing the gradients of the parameters. instead, use another "dummy forward" to get them again
    * consider two types of implementation. only use true NP when the create_fwd_hook_assign_grad has been implemented for that layer
      * pseudo NP: explicilty get the ZO gradient estimation of the output, still build and call BP, but use additional backward hook to replace the true gradients of the output with the ZO estimation
      * true NP: do it in a memory-efficient way: do not explicilty save the ZO gradient estimation of the output, regenerate based on saved random seeds and scaling_factors. do not create BP graph, but use customized forward hook, register before "dummy forward", and compute the gradients of parameters in the dummy forward when the input activation is again available

* general method
  * the gradient estimator should fit both 2-d activation (batch, dim) an