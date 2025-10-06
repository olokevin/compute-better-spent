ds=cifar10 # choose from {cifar10, cifar100}
lr=3e-3
depth=3
width=64
struct=btt
layers=all_but_last
# for scale_factor in 2 4 8 16 32 64 128 256; do
scale_factor=32

python3 train_cifar.py \
  --wandb_project=mlp_${ds} \
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
  --scheduler=cosine