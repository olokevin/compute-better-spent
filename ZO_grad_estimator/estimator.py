"""
Main ZO Gradient Estimator.

Supports:
- Weight perturbation (WP): Directly perturb parameters
- Node perturbation (NP): Perturb activations
  - Pseudo-NP: Use backward hooks to replace gradients
  - True-NP: Memory-efficient, uses forward hooks for direct grad computation
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any, Dict
from .config import ZOConfig
from .utils import (
    PerturbParam, PerturbLayer,
    build_random_generator,
    find_layers_by_rules, find_params_by_rules,
    create_fwd_hook_add_perturbation, create_fwd_hook_store_output_shape,
    split_model,
    compute_layer_dimensions, compute_spectral_sigma_wp, compute_spectral_sigma_np,
    compute_spectral_lr_mult_wp, compute_spectral_lr_mult_np
)
from .objective import ObjectiveFunction


class ZOEstimator(nn.Module):
    """
    Monte Carlo ZO Gradient Estimator.

    Memory-efficient implementation:
    - Never explicitly saves perturbations (regenerates from seeds)
    - Wraps all computation in torch.no_grad() when possible
    - Cleans up intermediate results to free GPU memory
    """

    def __init__(self, config: ZOConfig, model: nn.Module):
        super().__init__()

        self.config = config
        self.model = model
        self.device = next(model.parameters()).device

        # Build random generator
        self.rand_gen_fn = build_random_generator(config.sample_method, self.device)

        # Initialize perturbation lists
        self.perturb_params: List[PerturbParam] = []
        self.perturb_layers: List[PerturbLayer] = []

        # Select parameters/layers to perturb
        self._select_perturb_targets()

        # Objective function (set via update_objective)
        self.objective_fn: Optional[ObjectiveFunction] = None

        # Counter for tracking forward passes
        self.forward_counter = 0

    def _select_perturb_targets(self):
        """Select parameters or layers to perturb based on config rules."""

        # === Parameter Perturbation (Weight Perturbation) ===
        if self.config.param_perturb_rules is not None:
            print('\n=== Parameter Perturbation (Weight Perturbation) ===')
            matched_params = find_params_by_rules(
                self.model,
                self.config.param_perturb_rules,
                require_grad=True,
                verbose=True
            )

            for idx, (param_name, param, rule_name) in enumerate(matched_params):
                perturb_param = PerturbParam(idx=idx, name=param_name, param=param)

                # Compute spectral scaling if enabled
                if self.config.en_spectral_scaling:
                    n_in, n_out, d = compute_layer_dimensions(param)

                    # Compute layer-wise sigma
                    if self.config.spectral_sigma_method == 'wp_standard':
                        perturb_param.sigma = compute_spectral_sigma_wp(n_in, n_out, d, self.config.sigma)
                    else:
                        perturb_param.sigma = self.config.sigma

                    # Compute layer-wise LR multiplier
                    if self.config.spectral_lr_method == 'zo_variance_adjusted':
                        # Use layer-wise sigma if available
                        sigma_for_lr = perturb_param.sigma if perturb_param.sigma is not None else self.config.sigma
                        perturb_param.lr_mult = compute_spectral_lr_mult_wp(
                            n_in, n_out, d,
                            self.config.n_sample,
                            sigma_for_lr,
                            self.config.spectral_C_constant
                        )
                    else:
                        perturb_param.lr_mult = 1.0

                    print(f'    {param_name}: n_in={n_in}, n_out={n_out}, d={d}, '
                          f'sigma={perturb_param.sigma:.6f}, lr_mult={perturb_param.lr_mult:.4f}')
                else:
                    perturb_param.sigma = self.config.sigma
                    perturb_param.lr_mult = 1.0

                self.perturb_params.append(perturb_param)

        # === Activation Perturbation (Node Perturbation) ===
        if self.config.actv_perturb_rules is not None:
            print('\n=== Activation Perturbation (Node Perturbation) ===')
            matched_layers = find_layers_by_rules(
                self.model,
                self.config.actv_perturb_rules,
                require_grad=True,
                verbose=True
            )

            for idx, (layer_name, layer, rule_name) in enumerate(matched_layers):
                self.perturb_layers.append(
                    PerturbLayer(idx=idx, name=layer_name, layer=layer)
                )

            # Handle pseudo-ZO exclusions
            if self.config.en_pseudo_ZO:
                original_count = len(self.perturb_layers)
                self.perturb_layers = [
                    layer for layer in self.perturb_layers
                    if 'classifier' not in layer.name
                ]
                print(f'\nPseudo-ZO: Excluded {original_count - len(self.perturb_layers)} classifier layers')

        # Summary
        print(f'\n=== ZO Estimator Summary ===')
        print(f'Parameters to perturb: {len(self.perturb_params)}')
        print(f'Layers to perturb: {len(self.perturb_layers)}')
        print(f'Perturbation method: {"Weight" if self.perturb_params else "Node"}')

    def update_objective(self, objective_fn: ObjectiveFunction):
        """Update the objective function with new data."""
        self.objective_fn = objective_fn

    def get_lr_multipliers(self) -> Dict[str, float]:
        """
        Get learning rate multipliers for all parameters with spectral scaling.

        Returns:
            Dictionary mapping parameter names to LR multipliers
        """
        lr_mults = {}
        for perturb_param in self.perturb_params:
            if perturb_param.lr_mult is not None:
                lr_mults[perturb_param.name] = perturb_param.lr_mult
        return lr_mults

    def estimate_grad(self) -> Tuple[Any, torch.Tensor]:
        """
        Main entry point for gradient estimation.

        Returns:
            (output, loss) tuple from objective function
        """
        if self.objective_fn is None:
            raise ValueError("Objective function not set. Call update_objective() first.")

        # Dispatch to appropriate estimation method
        if self.perturb_params:
            return self._estimate_grad_weight_perturb()
        elif self.perturb_layers:
            return self._estimate_grad_node_perturb()
        else:
            raise ValueError("No parameters or layers selected for perturbation")

    def _estimate_grad_weight_perturb(self) -> Tuple[Any, torch.Tensor]:
        """Weight perturbation: Perturb parameters directly."""

        with torch.no_grad():
            # Get baseline loss
            _, old_loss = self.objective_fn(return_loss_reduction='none')
            batch_sz = old_loss.numel()

        # Check if layerwise perturbation is enabled
        if self.config.en_layerwise_perturbation:
            # Layerwise: Perturb one parameter at a time
            for i in range(self.config.n_sample):
                for perturb_param in self.perturb_params:
                    # Generate seed and perturbation
                    seed = torch.randint(0, 100000, (1,)).item()

                    state = torch.get_rng_state()
                    torch.manual_seed(seed)
                    perturbation = self.rand_gen_fn(perturb_param.param.shape)
                    torch.set_rng_state(state)

                    # Apply perturbation with layer-wise sigma
                    sigma = perturb_param.sigma if perturb_param.sigma is not None else self.config.sigma
                    with torch.no_grad():
                        perturb_param.param.add_(sigma * perturbation)

                    # Forward pass
                    with torch.no_grad():
                        _, pos_loss = self.objective_fn(return_loss_reduction='none')
                        self.forward_counter += 1

                    # Compute loss difference
                    if self.config.estimate_method == 'forward':
                        loss_diff = pos_loss - old_loss
                        # Restore parameter
                        with torch.no_grad():
                            perturb_param.param.sub_(sigma * perturbation)
                    elif self.config.estimate_method == 'antithetic':
                        # Remove positive, add negative
                        with torch.no_grad():
                            perturb_param.param.sub_(2 * sigma * perturbation)

                        with torch.no_grad():
                            _, neg_loss = self.objective_fn(return_loss_reduction='none')
                            self.forward_counter += 1

                        loss_diff = (pos_loss - neg_loss) / 2.0

                        # Restore to original
                        with torch.no_grad():
                            perturb_param.param.add_(sigma * perturbation)
                    else:
                        raise ValueError(f"Unknown estimate_method: {self.config.estimate_method}")

                    # Gradient estimate: g ≈ (1/σ) * Δloss * u
                    grad_est = (loss_diff.mean() / sigma) * perturbation

                    if perturb_param.param.grad is None:
                        perturb_param.param.grad = grad_est
                    else:
                        perturb_param.param.grad += grad_est

        else:
            # Simultaneous perturbation: Perturb all parameters at once
            for i in range(self.config.n_sample):
                # Generate and apply perturbations
                seeds = []
                for perturb_param in self.perturb_params:
                    seed = torch.randint(0, 100000, (1,)).item()
                    seeds.append(seed)

                    # Generate perturbation
                    state = torch.get_rng_state()
                    torch.manual_seed(seed)
                    perturbation = self.rand_gen_fn(perturb_param.param.shape)
                    torch.set_rng_state(state)

                    # Apply perturbation with layer-wise sigma
                    sigma = perturb_param.sigma if perturb_param.sigma is not None else self.config.sigma
                    with torch.no_grad():
                        perturb_param.param.add_(sigma * perturbation)

                # Forward pass
                with torch.no_grad():
                    _, pos_loss = self.objective_fn(return_loss_reduction='none')
                    self.forward_counter += 1

                # Compute loss difference
                if self.config.estimate_method == 'forward':
                    loss_diff = pos_loss - old_loss
                elif self.config.estimate_method == 'antithetic':
                    # Remove positive perturbation, add negative
                    for perturb_param in self.perturb_params:
                        state = torch.get_rng_state()
                        torch.manual_seed(seeds[perturb_param.idx])
                        perturbation = self.rand_gen_fn(perturb_param.param.shape)
                        torch.set_rng_state(state)

                        sigma = perturb_param.sigma if perturb_param.sigma is not None else self.config.sigma
                        with torch.no_grad():
                            perturb_param.param.sub_(2 * sigma * perturbation)

                    with torch.no_grad():
                        _, neg_loss = self.objective_fn(return_loss_reduction='none')
                        self.forward_counter += 1

                    loss_diff = (pos_loss - neg_loss) / 2.0

                # Remove perturbations
                for perturb_param in self.perturb_params:
                    state = torch.get_rng_state()
                    torch.manual_seed(seeds[perturb_param.idx])
                    perturbation = self.rand_gen_fn(perturb_param.param.shape)
                    torch.set_rng_state(state)

                    sigma = perturb_param.sigma if perturb_param.sigma is not None else self.config.sigma
                    sign = 1 if self.config.estimate_method == 'forward' else -1
                    with torch.no_grad():
                        perturb_param.param.sub_(sign * sigma * perturbation)

                # Accumulate gradient
                for perturb_param in self.perturb_params:
                    state = torch.get_rng_state()
                    torch.manual_seed(seeds[perturb_param.idx])
                    perturbation = self.rand_gen_fn(perturb_param.param.shape)
                    torch.set_rng_state(state)

                    # Gradient estimate: g ≈ (1/σ) * Δloss * u
                    sigma = perturb_param.sigma if perturb_param.sigma is not None else self.config.sigma
                    grad_est = (loss_diff.mean() / sigma) * perturbation

                    if perturb_param.param.grad is None:
                        perturb_param.param.grad = grad_est
                    else:
                        perturb_param.param.grad += grad_est

        # Scale gradients
        scaling_factor = 1.0 / self.config.n_sample
        for perturb_param in self.perturb_params:
            if perturb_param.param.grad is not None:
                perturb_param.param.grad.mul_(scaling_factor)

                if self.config.signsgd:
                    perturb_param.param.grad = torch.sign(perturb_param.param.grad)

        # Final forward pass for metrics
        output, loss = self.objective_fn(return_loss_reduction='mean')
        return output, loss

    def _estimate_grad_node_perturb(self) -> Tuple[Any, torch.Tensor]:
        """
        Node perturbation: Perturb activations.

        Returns ZO gradient of activations (ZO_grad_output) which is then
        used with backward hooks (pseudo-NP) or forward hooks (true-NP).
        """

        with torch.no_grad():
            # Register hooks to store output shapes
            hooks = []
            for perturb_layer in self.perturb_layers:
                hook = perturb_layer.layer.register_forward_hook(create_fwd_hook_store_output_shape)
                hooks.append(hook)

            # Baseline forward to get shapes and loss
            _, old_loss = self.objective_fn(return_loss_reduction='none')
            batch_sz = old_loss.shape[0] if old_loss.dim() > 0 else 1

            # Remove hooks
            for hook in hooks:
                hook.remove()

        # Perturbation loop
        scaling_factor = 1.0 / (self.config.n_sample * batch_sz)

        for i in range(self.config.n_sample):
            # Assign random seeds to each layer
            for perturb_layer in self.perturb_layers:
                seed = torch.randint(0, 100000, (1,)).item()
                perturb_layer.seed_list.append(seed)

            # Positive perturbation forward
            with torch.no_grad():
                fwd_hooks = []
                for perturb_layer in self.perturb_layers:
                    hook = perturb_layer.layer.register_forward_hook(
                        create_fwd_hook_add_perturbation(
                            seed=perturb_layer.seed_list[i],
                            sigma=self.config.sigma,
                            rand_gen_fn=self.rand_gen_fn,
                            mask=None
                        )
                    )
                    fwd_hooks.append(hook)

                _, pos_loss = self.objective_fn(return_loss_reduction='none')
                self.forward_counter += 1

                for hook in fwd_hooks:
                    hook.remove()

            # Compute loss difference
            if self.config.estimate_method == 'forward':
                loss_diff = pos_loss - old_loss
            elif self.config.estimate_method == 'antithetic':
                with torch.no_grad():
                    fwd_hooks = []
                    for perturb_layer in self.perturb_layers:
                        hook = perturb_layer.layer.register_forward_hook(
                            create_fwd_hook_add_perturbation(
                                seed=perturb_layer.seed_list[i],
                                sigma=-self.config.sigma,
                                rand_gen_fn=self.rand_gen_fn,
                                mask=None
                            )
                        )
                        fwd_hooks.append(hook)

                    _, neg_loss = self.objective_fn(return_loss_reduction='none')
                    self.forward_counter += 1

                    for hook in fwd_hooks:
                        hook.remove()

                loss_diff = (pos_loss - neg_loss) / 2.0

            # Store scale factors for each layer
            for perturb_layer in self.perturb_layers:
                perturb_layer.scale_factor_list.append(scaling_factor * loss_diff / self.config.sigma)

        # Generate ZO_grad_output for each layer (pseudo-NP)
        for perturb_layer in self.perturb_layers:
            output_shape = perturb_layer.layer.output_shape

            ZO_grad_output = None
            for seed, scale_factor in zip(perturb_layer.seed_list, perturb_layer.scale_factor_list):
                state = torch.get_rng_state()
                torch.manual_seed(seed)
                perturbation = self.rand_gen_fn(output_shape)
                torch.set_rng_state(state)

                # Handle different scale_factor shapes (batch-wise or sequence-wise)
                if scale_factor.dim() > 0:
                    scale_factor = scale_factor.unsqueeze(-1)

                if ZO_grad_output is None:
                    ZO_grad_output = scale_factor * perturbation
                else:
                    ZO_grad_output += scale_factor * perturbation

            # Apply sign SGD if requested
            if self.config.signsgd:
                ZO_grad_output = torch.sign(ZO_grad_output)

            # Store for backward hook
            perturb_layer.layer.ZO_grad_output = ZO_grad_output

        # This returns without computing final output
        # The training loop will handle pseudo-NP (backward hooks) or true-NP (forward hooks)
        return None, None
