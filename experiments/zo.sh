ds=cifar10 # choose from {cifar10, cifar100}
lr=3e-3
depth=3
width=64
struct=btt
layers=all_but_last
# for scale_factor in 2 4 8 16 32 64 128 256; do
scale_factor=32

run_dense_zo_np(){
  ds=cifar10 # choose from {cifar10, cifar100}
  lr=3e-3
  depth=3
  width=64
  struct=dense
  wandb_name_append=zo_np_${time_stamp}
  # for scale_factor in 0.5 1 2 4 8 16 32 64; do
  for scale_factor in 1; do
  python3 train_cifar.py \
  --wandb_project=zo_mlp_${ds} \
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
  --scheduler=cosine \
  --wandb_name_append=${wandb_name_append} \
  --ZO_config_path=ZO_grad_estimator/config/zo_cifar_mlp_np.yaml
  done;
}

run_dense_zo_wp(){
  ds=cifar10 # choose from {cifar10, cifar100}
  lr=3e-3
  depth=3
  width=64
  struct=dense
  wandb_name_append=zo_wp_${time_stamp}
  # for scale_factor in 0.5 1 2 4 8 16 32 64; do
  for scale_factor in 1; do
  python3 train_cifar.py \
  --wandb_project=zo_mlp_${ds} \
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
  --scheduler=cosine \
  --wandb_name_append=${wandb_name_append} \
  --ZO_config_path=ZO_grad_estimator/config/zo_cifar_mlp_wp.yaml
  done;
}

run_dense_zo_np
# run_dense_zo_wp

# --wandb_name_override=checkpoints/${ds}/zo_wp/${struct}-${depth}-$((scale_factor*width))-${lr}-${time_stamp} \