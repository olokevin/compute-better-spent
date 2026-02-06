ds=cifar10 # choose from {cifar10, cifar100}
time_stamp=$(date +%Y%m%d_%H%M%S)

# Common config for all structures
depth=3
width=64
scale_factor=16  # Fixed scale factor for fair comparison
layers=all_but_last

# Calculate tt_rank for BTT structures. √d/2: ensures same parameters
tt_rank=$(echo "sqrt(${width}*${scale_factor})" | bc -l)
tt_rank=$(printf "%.0f" "$tt_rank")

btt_rank=$(echo "sqrt(${width}*${scale_factor})/2" | bc -l)
btt_rank=$(printf "%.0f" "$btt_rank")

wandb_name_append=cfg-

# optimizer=adamw
# lr=3e-3 # for width=64

optimizer=muon
lr=3e-3
# wandb_name_append=$wandb_name_append-rms_default

wandb_project=struct_mlp_muon

# Common training args
common_args="--wandb_project=${wandb_project} \
  --dataset=${ds} \
  --model=MLP \
  --width=${width} \
  --depth=${depth} \
  --batch_size=1024 \
  --epochs=500 \
  --resolution=32 \
  --optimizer=${optimizer} \
  --lr=${lr} \
  --scale_factor=${scale_factor} \
  --input_lr_mult=0.1 \
  --scheduler=cosine \
  --wandb_name_append=${wandb_name_append}"

  # --use_wrong_mult \


# Low Rank

run_low_rank(){
python3 train_cifar.py ${common_args} \
  --struct=low_rank \
  --layers=${layers}
}

# Low Rank with Activation
run_low_rank_actv(){
python3 train_cifar.py ${common_args} \
  --struct=low_rank_actv \
  --layers=${layers} \
  --low_rank_activation=gelu 
}

# BTT with Activation
run_btt_actv(){
python3 train_cifar.py ${common_args} \
  --struct=btt_actv \
  --layers=${layers} \
  --tt_rank=${btt_rank} \
  --low_rank_activation=gelu 
}



# Dense
run_dense(){
python3 train_cifar.py ${common_args} \
  --struct=dense \
  --muon_enable_mup_retraction
}

# TT
run_tt(){
python3 train_cifar.py ${common_args} \
  --struct=tt \
  --layers=${layers} \
  --tt_rank=${tt_rank} \
  --muon_structured_ortho_method=muP \
  --muon_structured_adjust_lr_method=muP 
}

# BTT
run_btt(){
python3 train_cifar.py ${common_args} \
  --struct=btt \
  --layers=${layers} \
  --tt_rank=${btt_rank} \
  --init_method=mup_btt \
  --muon_structured_ortho_method=mup_btt \
  --muon_structured_adjust_lr_method=mup_btt \
  --decomp_mode=input_one_block
}


# BTT LR search (adamw + muon)
run_lr_search(){
  base_args="--wandb_project=${wandb_project} \
  --dataset=${ds} \
  --model=MLP \
  --width=${width} \
  --depth=${depth} \
  --batch_size=1024 \
  --epochs=500 \
  --resolution=32 \
  --scale_factor=${scale_factor} \
  --input_lr_mult=0.1 \
  --weight_decay=0.01 \
  --scheduler=cosine 
  "

  # optimizer=adamw
  # lrs="3e-4 1e-3 3e-3 1e-2"

  optimizer=muon
  # lrs="1e-2 3e-2 1e-1"

  # lrs="3e-4 3e-3 3e-2"
  # lrs="1e-3 1e-2 1e-1"

  lrs="1e-3 3e-3 1e-2 3e-2"
  # lrs="3e-4 3e-3 3e-2 1e-3 1e-2 1e-1"

  for lr in ${lrs}; do
    wandb_name_append="wd0_01-output_one_block-muP-mup_btt-mup_btt"

    # python3 train_cifar.py ${base_args} \
    #   --optimizer=${optimizer} \
    #   --lr=${lr} \
    #   --wandb_name_append=${wandb_name_append} \
    #   --struct=dense \
    #   --layers=${layers} \
    #   --muon_enable_mup_retraction \
    # >/dev/null 2>&1 &
    
    # python3 train_cifar.py ${base_args} \
    #   --optimizer=${optimizer} \
    #   --lr=${lr} \
    #   --wandb_name_append=${wandb_name_append} \
    #   --struct=tt \
    #   --layers=${layers} \
    #   --tt_rank=${tt_rank} \
    #   --muon_structured_ortho_method=muP \
    #   --muon_structured_adjust_lr_method=muP \
    # >/dev/null 2>&1 &

    python3 train_cifar.py ${base_args} \
      --optimizer=${optimizer} \
      --lr=${lr} \
      --wandb_name_append=${wandb_name_append} \
      --struct=btt \
      --layers=${layers} \
      --tt_rank=${btt_rank} \
      --decomp_mode=output_one_block \
      --init_method=µP \
      --muon_structured_ortho_method=mup_btt \
      --muon_structured_adjust_lr_method=mup_btt \
    >/dev/null 2>&1 &
  done
}

# --decomp_mode=input_one_block \  # input_one_block output_one_block
# --init_method=mup_btt \
# --muon_structured_ortho_method=mup_btt \
# --muon_structured_adjust_lr_method=mup_btt \  # mup_btt_new

# --init_method=µP \
# --muon_structured_ortho_method=muP \
# --muon_structured_adjust_lr_method=muP \

# bash experiments/muon.sh >/dev/null 2>&1 &

export CUDA_VISIBLE_DEVICES=4


# run_low_rank
# run_low_rank_actv
# run_btt_actv

# run_dense
# run_tt
# run_btt
run_lr_search


