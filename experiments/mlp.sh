export CUDA_VISIBLE_DEVICES=2
ds=cifar10 # choose from {cifar10, cifar100}
lr=3e-3

### Dense ####
run_dense(){
  depth=3
  width=64
  struct=dense
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
  --scheduler=cosine
  done;
}

### Kron ####
run_kron(){ 
depth=3
width=64
struct=kron
layers=all_but_last
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
--scheduler=cosine
done; 
}

### Low Rank ####
run_low_rank(){
depth=3
width=64
struct=low_rank
layers=all_but_last
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
--scheduler=cosine
done;
}

### Low Rank with Activation ####
run_low_rank_actv(){
depth=3
width=64
struct=low_rank_actv
layers=all_but_last
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
--activation=gelu \
--scheduler=cosine
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
# tt_rank=1
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
--tt_rank=${tt_rank}
done;
}

### BTT with Activation ####
run_btt_actv(){
depth=3
width=64
struct=btt_actv
layers=all_but_last
for scale_factor in 2 4 8 16 32 64 128 256; do
tt_rank=$(echo "sqrt(${width}*${scale_factor})/2" | bc -l)
tt_rank=$(printf "%.0f" "$tt_rank")
# tt_rank=1
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
--activation=relu \
--scheduler=cosine \
--tt_rank=${tt_rank}
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

# run_dense
# run_kron
# run_monarch
# run_tt

run_low_rank
# run_low_rank_actv
# run_btt
# run_btt_actv

# nohup bash experiments/mlp.sh >/dev/null 2>&1 &