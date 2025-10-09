"""
Integration Example: Using ZO Gradient Estimator with CIFAR Training

This shows how to integrate the new ZO_grad_estimator into train_cifar.py
"""

import torch
from torch.nn import CrossEntropyLoss
from ZO_grad_estimator import ZOEstimator, ZOConfig, build_objective_function
from ZO_grad_estimator.utils import create_bwd_pre_hook_replace_grad


def train_step_with_zo(model, optimizer, ims, targs, loss_fn, zo_estimator, args):
    """
    Single training step with ZO gradient estimation.

    Args:
        model: Neural network model
        optimizer: PyTorch optimizer
        ims: Input images
        targs: Target labels
        loss_fn: Loss function
        zo_estimator: ZOEstimator instance
        args: Training arguments

    Returns:
        (output, loss) tuple
    """

    # Build objective function
    obj_fn = build_objective_function(
        obj_fn_type='CIFAR',
        model=model,
        ims=ims,
        targs=targs,
        loss_fn=loss_fn,
        args=args
    )

    # Update ZO estimator with new objective
    zo_estimator.update_objective(obj_fn)

    # === WEIGHT PERTURBATION ===
    if zo_estimator.perturb_params:
        # Set model to eval (disable dropout for ZO)
        model.eval()

        # Estimate gradients (assigns to param.grad)
        output, loss = zo_estimator.estimate_grad()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        return output, loss

    # === NODE PERTURBATION (Pseudo-NP with backward hooks) ===
    elif zo_estimator.perturb_layers:
        # Step 1: Estimate ZO_grad_output (with model in eval mode)
        model.eval()
        zo_estimator.estimate_grad()

        # Step 2: Use backward hooks to get param gradients
        model.train()
        bwd_hooks = []

        for perturb_layer in zo_estimator.perturb_layers:
            hook = perturb_layer.layer.register_full_backward_pre_hook(
                create_bwd_pre_hook_replace_grad(
                    perturb_layer.layer.ZO_grad_output,
                    debug=False
                )
            )
            bwd_hooks.append(hook)

        # Forward and backward
        output, loss = obj_fn()
        loss.backward()

        # Remove hooks
        for hook in bwd_hooks:
            hook.remove()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        return output, loss

    else:
        raise ValueError("No perturbation method selected")


# ===== Full Training Loop Example =====

def train_epoch_with_zo(model, optimizer, scheduler, loss_fn, train_loader, zo_estimator, args):
    """Full training epoch with ZO."""
    from scaling_mlps.utils.metrics import topk_acc, AverageMeter
    import time

    start = time.time()
    model.train()

    total_acc = AverageMeter()
    total_loss = AverageMeter()

    for step, (ims, targs) in enumerate(train_loader):
        # ZO training step
        output, loss = train_step_with_zo(
            model, optimizer, ims, targs, loss_fn, zo_estimator, args
        )

        # Compute accuracy
        preds = output.logits
        targs_for_acc = targs[:, 0].long() if args.mixup > 0 else targs
        acc, _ = topk_acc(preds, targs_for_acc, None, k=5, avg=True)

        total_acc.update(acc, ims.shape[0])
        total_loss.update(loss.item(), ims.shape[0])

    scheduler.step()
    end = time.time()

    return total_acc.get_avg(percentage=True), total_loss.get_avg(), end - start


# ===== Initialization Example =====

def setup_zo_estimator(model, config_path):
    """
    Initialize ZO estimator from config file.

    Args:
        model: PyTorch model
        config_path: Path to YAML config file

    Returns:
        ZOEstimator instance
    """
    # Load config
    config = ZOConfig.from_yaml(config_path)

    # Create estimator
    zo_estimator = ZOEstimator(config, model)

    print(f"\n=== ZO Estimator Initialized ===")
    print(f"Perturbation method: {'Weight' if zo_estimator.perturb_params else 'Node'}")
    print(f"Sample method: {config.sample_method}")
    print(f"Estimate method: {config.estimate_method}")
    print(f"Sigma: {config.sigma}")
    print(f"N samples: {config.n_sample}")

    return zo_estimator


# ===== Usage in main() =====

def main_with_zo(args):
    """Example main function with ZO training."""
    # ... (model setup, data loading, etc.)

    # Initialize ZO estimator
    zo_estimator = None
    if args.ZO_config_path is not None:
        zo_estimator = setup_zo_estimator(model, args.ZO_config_path)

    # Training loop
    for epoch in range(args.epochs):
        if zo_estimator is not None:
            # Train with ZO
            train_acc, train_loss, train_time = train_epoch_with_zo(
                model, optimizer, scheduler, loss_fn,
                train_loader, zo_estimator, args
            )
        else:
            # Standard training
            train_acc, train_loss, train_time = train_epoch_standard(
                model, optimizer, scheduler, loss_fn,
                train_loader, args
            )

        # ... (evaluation, logging, etc.)


if __name__ == "__main__":
    """
    To use this with train_cifar.py:

    1. Replace the ZO_Estim initialization with:
       from ZO_grad_estimator import ZOEstimator, ZOConfig
       zo_estimator = setup_zo_estimator(model, args.ZO_config_path)

    2. Replace the training loop ZO logic with:
       output, loss = train_step_with_zo(
           model, opt, ims, targs, loss_fn, zo_estimator, args
       )

    3. Run:
       python train_cifar.py --dataset=cifar10 --model=MLP \
         --width=64 --depth=3 --struct=btt --layers=all_but_last \
         --ZO_config_path=ZO_grad_estimator/config_examples/zo_cifar_mlp.yaml
    """
    print(__doc__)
