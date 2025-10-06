#!/bin/bash
# Experiment script for btt_actv on CIFAR-10/100 with MLP

ds=cifar10 # choose from {cifar10, cifar100}
lr=3e-3

### BTT with Activation ####
depth=3
width=64
struct=btt_actv
layers=all_but_last
activation=relu  # can be relu, gelu, silu, tanh

echo "Running btt_actv experiments on ${ds}"
echo "Using activation: ${activation}"

for scale_factor in 2 4 8 16 32 64 128 256; do
    echo "Running with scale_factor=${scale_factor}"
    python3 train_cifar.py \
        --wandb_project=mlp_${ds}_btt_actv \
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
