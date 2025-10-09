#!/bin/bash
# Test script for DEBUG and OUT_GRAD_DEBUG modes

# Activate environment
conda activate struct

echo "=== Testing DEBUG Mode (Weight Perturbation) ==="
echo "This will compare ZO parameter gradients with true BP gradients"
echo ""

# Test WP with DEBUG (requires setting DEBUG=True in train_cifar.py)
python train_cifar.py \
  --dataset=cifar10 --model=MLP --width=64 --depth=3 \
  --struct=btt --layers=all_but_last \
  --batch_size=128 --epochs=1 \
  --ZO_config_path=ZO_grad_estimator/config/zo_cifar_mlp_wp.yaml

echo ""
echo "=== Testing OUT_GRAD_DEBUG + DEBUG Modes (Node Perturbation) ==="
echo "This will compare both ZO output gradients and parameter gradients with true BP"
echo "(Requires setting DEBUG=True and OUT_GRAD_DEBUG=True in train_cifar.py)"
echo ""

# Test NP with both debug modes
python train_cifar.py \
  --dataset=cifar10 --model=MLP --width=64 --depth=3 \
  --struct=btt --layers=all_but_last \
  --batch_size=128 --epochs=1 \
  --ZO_config_path=ZO_grad_estimator/config/zo_cifar_mlp.yaml

echo ""
echo "=== Test Complete ==="
echo "Check the output above for cosine similarity values"
echo "Expected results:"
echo "  - WP: cos_sim > 0.5 (moderate correlation)"
echo "  - NP output grads: cos_sim > 0.7 (strong correlation)"
echo "  - NP param grads: cos_sim > 0.7 (strong correlation)"
