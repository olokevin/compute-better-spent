#!/bin/bash
# Experiment script for low_rank_actv on CIFAR-10/100 with MLP

ds=cifar10 # choose from {cifar10, cifar100}
lr=3e-3

### Low Rank with Activation ####
depth=3
width=64
struct=low_rank_actv
layers=all_but_last
activation=gelu  # can be gelu, relu, silu, tanh

echo "Running low_rank_actv experiments on ${ds}"
echo "Using activation: ${activation}"

for scale_factor in 2 4 8 16 32 64 128 256; do
    echo "Running with scale_factor=${scale_factor}"
    python3 train_cifar.py \
        --wandb_project=mlp_${ds}_low_rank_actv \
        --dataset=${ds} \
        --model=MLP \
        --width=${width} \
        --depth=${depth} \
        --lr=${lr} \
        --batch_size=1024 \
        --epochs=500 \
        --resolution=32 \
        --optimizer=adamw \
        --scale_factor=${scale_factor} \
        --input_lr_mult=0.1 \
        --struct=${struct} \
        --layers=${layers} \
        --activation=${activation} \
        --scheduler=cosine
done

echo "Experiments completed!"
