"""
Helper functions for weight perturbation with spectral scaling.
This module contains the updated _estimate_grad_weight_perturb method with layer-wise sigma support.
"""

def get_layer_sigma(perturb_param, config_sigma):
    """Get the appropriate sigma for a parameter (layer-wise or global)."""
    return perturb_param.sigma if perturb_param.sigma is not None else config_sigma


# Updated layerwise perturbation section (lines 168-219 of estimator.py)
LAYERWISE_PERTURBATION_CODE = """
        # Check if layerwise perturbation is enabled
        if self.config.en_layerwise_perturbation:
            # Layerwise: Perturb one parameter at a time
            for _ in range(self.config.n_sample):
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
"""

# Updated simultaneous perturbation section (lines 221-289 of estimator.py)
SIMULTANEOUS_PERTURBATION_CODE = """
        else:
            # Simultaneous perturbation: Perturb all parameters at once
            for _ in range(self.config.n_sample):
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
"""
