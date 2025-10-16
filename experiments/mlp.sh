export CUDA_VISIBLE_DEVICES=3
ds=cifar10 # choose from {cifar10, cifar100}
# lr=3e-3
lr=3e-1
time_stamp=$(date +%Y%m%d_%H%M%S)

### Dense ####
run_dense(){
  depth=3
  width=64
  struct=dense
  wandb_name_append=${time_stamp}
  for scale_factor in 0.5 1 2 4 8 16 32 64; do
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
  --scheduler=cosine \
  --wandb_name_append=${wandb_name_append} 
  done;
}

### Kron ####
run_kron(){ 
depth=3
width=64
struct=kron
layers=all_but_last
wandb_name_append=${time_stamp}
for scale_factor in 2 4 8 16 32 32 64 128 256; do
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
--scheduler=cosine \
--wandb_name_append=${wandb_name_append}
done;
}

### Monarch ####
run_monarch(){
depth=3
width=64
struct=monarch
layers=all_but_last
for scale_factor in 0.7 1.4 2.8 5.6 11 22 45 90; do
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
done;
}

### TT ####
run_tt(){
depth=3
width=64
struct=tt
layers=all_but_last
for scale_factor in 0.25 0.5 1 2 4 8 16 32 64; do
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
--tt_rank=16 \
--scheduler=cosine
done;
}

### Low Rank ####
run_low_rank(){
depth=3
width=64
struct=low_rank
layers=all_but_last
wandb_name_append=${time_stamp}
for scale_factor in 2 4 8 16 32 64 128 256; do
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
--scheduler=cosine \
--wandb_name_append=${wandb_name_append}
done;
}

### Low Rank with Activation ####
run_low_rank_actv(){
depth=3
width=64
struct=low_rank_actv
layers=all_but_last
wandb_name_append=low_rank_actv_gelu-mlp_actv_none-${time_stamp}
for scale_factor in 2 4 8 16 32 64 128 256; do
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
--scheduler=cosine \
--low_rank_activation=gelu \
--mlp_activation=none \
--wandb_name_append=${wandb_name_append}
done;
}

### BTT ####
run_btt(){
depth=3
width=64
struct=btt
layers=all_but_last
for scale_factor in 2 4 8 16 32 64 128 256; do
tt_rank=$(echo "sqrt(${width}*${scale_factor})/2" | bc -l)
tt_rank=$(printf "%.0f" "$tt_rank")
wandb_name_append=rank_sqrt_${time_stamp}
# tt_rank=1
# wandb_name_append=rank1
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
--scheduler=cosine \
--wandb_name_append=${wandb_name_append}
--tt_rank=${tt_rank}
done;
}

### BTT with Activation ####
run_btt_actv(){
  depth=3
  width=64
  struct=btt_actv
  layers=all_but_last
  # for scale_factor in 2 4 8 16 32 64 128 256; do
  for scale_factor in 2; do
  tt_rank=$(echo "sqrt(${width}*${scale_factor})/2" | bc -l)
  tt_rank=$(printf "%.0f" "$tt_rank")
  wandb_name_append=rank_sqrt-low_rank_actv_gelu-mlp_actv_none-${time_stamp}
  # tt_rank=1
  # wandb_name_append=rank1
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
  --scheduler=cosine \
  --low_rank_activation=gelu \
  --mlp_activation=gelu \
  --tt_rank=${tt_rank} \
  --wandb_name_append=${wandb_name_append}
done;
}

### Compare Structures at Fixed Scale Factor ####
# Purpose: Compare different structures (dense, low_rank, low_rank_actv, btt, btt_actv)
#          using the SAME config at a FIXED scale_factor for direct comparison
# All runs use:
#   - Same scale_factor (default: 16)
#   - Same hyperparameters (lr, batch_size, epochs, etc.)
#   - Same wandb_name_append tag to group runs together
# Struct-specific configs:
#   - tt_rank for BTT structures (calculated from scale_factor)
#   - low_rank_activation=gelu, mlp_activation=none for *_actv variants
#   - layers=all_but_last for structured matrices (not dense)
run_compare(){
  # Common config for all structures
  depth=3
  width=64
  scale_factor=16  # Fixed scale factor for fair comparison
  layers=all_but_last
  wandb_name_append="compare-$((scale_factor*width))-uniform_lr-${time_stamp}"

  # Common training args
  common_args="--wandb_project=mlp_${ds} \
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
    --scheduler=cosine \
    --use_wrong_mult \
    --wandb_name_append=${wandb_name_append}"

  # Calculate tt_rank for BTT structures
  tt_rank=$(echo "sqrt(${width}*${scale_factor})/2" | bc -l)
  tt_rank=$(printf "%.0f" "$tt_rank")

  echo "=================================="
  echo "Running Structure Comparison"
  echo "Scale Factor: ${scale_factor}"
  echo "TT Rank (for BTT): ${tt_rank}"
  echo "=================================="

  # 1. Dense
  echo "Training: Dense"
  python3 train_cifar.py ${common_args} \
    --struct=dense

  # 2. Low Rank
  echo "Training: Low Rank"
  python3 train_cifar.py ${common_args} \
    --struct=low_rank \
    --layers=${layers}

  # 3. Low Rank with Activation
  echo "Training: Low Rank with Activation"
  python3 train_cifar.py ${common_args} \
    --struct=low_rank_actv \
    --layers=${layers} \
    --low_rank_activation=gelu 

  # 4. BTT
  echo "Training: BTT"
  python3 train_cifar.py ${common_args} \
    --struct=btt \
    --layers=${layers} \
    --tt_rank=${tt_rank}

  # 5. BTT with Activation
  echo "Training: BTT with Activation"
  python3 train_cifar.py ${common_args} \
    --struct=btt_actv \
    --layers=${layers} \
    --tt_rank=${tt_rank} \
    --low_rank_activation=gelu 

  echo "=================================="
  echo "Structure comparison completed!"
  echo "=================================="
}

# Choose which experiment to run:

# Individual structure scaling laws (vary scale_factor):
# run_dense
# run_kron
# run_monarch
# run_tt
# run_low_rank
# run_low_rank_actv
# run_btt
# run_btt_actv

# Zeroth-order training:
# run_dense_zo

# Structure comparison at fixed scale factor:
run_compare

# Run in background:
# nohup bash experiments/mlp.sh >/dev/null 2>&1 &