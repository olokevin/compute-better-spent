"""
Utility functions and classes for ZO gradient estimation.

Includes:
- Random perturbation generation
- Layer/parameter selection by rules
- Memory-efficient hooks
"""

import re
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Callable, Optional
from dataclasses import dataclass


# ==================== Data Structures ====================

@dataclass
class PerturbParam:
    """Represents a parameter to be perturbed."""
    idx: int
    name: str
    param: nn.Parameter
    layer: Optional[nn.Module] = None

    # Spectral scaling attributes
    sigma: Optional[float] = None  # Layer-wise perturbation scale
    lr_mult: Optional[float] = None  # Layer-wise learning rate multiplier


@dataclass
class PerturbLayer:
    """Represents a layer whose activations will be perturbed."""
    idx: int
    name: str
    layer: nn.Module
    mode: str = 'actv'  # 'actv' for activation perturbation, 'param' for parameter perturbation

    # For memory-efficient gradient computation
    seed_list: List[int] = None
    scale_factor_list: List[torch.Tensor] = None

    # Spectral scaling attributes
    sigma: Optional[float] = None  # Layer-wise perturbation scale
    lr_mult: Optional[float] = None  # Layer-wise learning rate multiplier

    def __post_init__(self):
        if self.seed_list is None:
            self.seed_list = []
        if self.scale_factor_list is None:
            self.scale_factor_list = []


# ==================== Random Perturbation Generation ====================

def build_random_generator(sample_method: str, device: torch.device) -> Callable:
    """Build random perturbation generator based on sampling method."""

    def gaussian_gen(shape):
        return torch.randn(shape, device=device)

    def bernoulli_gen(shape):
        return torch.ones(shape, device=device) - 2 * torch.bernoulli(0.5 * torch.ones(shape, device=device))

    def uniform_gen(shape):
        sample = torch.randn(shape, device=device)
        return torch.nn.functional.normalize(sample, p=2, dim=-1)

    generators = {
        'gaussian': gaussian_gen,
        'bernoulli': bernoulli_gen,
        'uniform': uniform_gen,
    }

    if sample_method not in generators:
        raise ValueError(f"Unknown sample_method: {sample_method}. Choose from {list(generators.keys())}")

    return generators[sample_method]


# ==================== Layer/Parameter Selection ====================

def get_type_mapping() -> Dict[str, type]:
    """Map string type names to actual class types."""
    try:
        from nn.cola_nn import CoLALayer
    except ImportError:
        CoLALayer = type(None)

    return {
        'CoLALayer': CoLALayer,
        'nn.Linear': nn.Linear,
        'nn.Conv2d': nn.Conv2d,
        'nn.LayerNorm': nn.LayerNorm,
        'nn.BatchNorm2d': nn.BatchNorm2d,
    }


def resolve_type_strings(rule_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string type names to actual classes."""
    if 'type' not in rule_spec:
        return rule_spec

    type_map = get_type_mapping()
    types = rule_spec['type']

    if isinstance(types, str):
        types = [types]

    resolved_types = []
    for t in types:
        if isinstance(t, str):
            if t in type_map:
                resolved_types.append(type_map[t])
            else:
                print(f"Warning: Unknown type '{t}', skipping")
        else:
            resolved_types.append(t)

    resolved_spec = rule_spec.copy()
    resolved_spec['type'] = resolved_types
    return resolved_spec


def match_layer_by_rules(layer_name: str, layer: nn.Module, rules: Dict[str, Dict]) -> Tuple[bool, Optional[str]]:
    """Check if a layer matches any selection rule."""
    for rule_name, rule_spec in rules.items():
        resolved_spec = resolve_type_strings(rule_spec)

        # Check name pattern
        name_pattern = resolved_spec.get('name_pattern', '.*')
        if not re.match(name_pattern, layer_name):
            continue

        # Check type if specified
        if 'type' in resolved_spec:
            allowed_types = resolved_spec['type']
            if not isinstance(allowed_types, (list, tuple)):
                allowed_types = [allowed_types]
            if not any(isinstance(layer, t) for t in allowed_types):
                continue

        return True, rule_name

    return False, None


def find_layers_by_rules(model: nn.Module, rules: Dict[str, Dict],
                         require_grad: bool = True, verbose: bool = True) -> List[Tuple[str, nn.Module, str]]:
    """Find all layers matching the rules."""
    matched_layers = []

    for layer_name, layer in model.named_modules():
        if not layer_name:
            continue

        matched, rule_name = match_layer_by_rules(layer_name, layer, rules)

        if matched:
            if require_grad:
                has_trainable = any(p.requires_grad for p in layer.parameters())
                if not has_trainable:
                    continue

            matched_layers.append((layer_name, layer, rule_name))
            if verbose:
                print(f"  Matched: {layer_name} ({type(layer).__name__}) [rule: {rule_name}]")

    return matched_layers


def find_params_by_rules(model: nn.Module, rules: Dict[str, Dict],
                         require_grad: bool = True, verbose: bool = True) -> List[Tuple[str, nn.Parameter, str]]:
    """Find all parameters matching the rules."""
    matched_params = []

    for param_name, param in model.named_parameters():
        if require_grad and not param.requires_grad:
            continue

        matched = False
        matched_rule = None

        for rule_name, rule_spec in rules.items():
            name_pattern = rule_spec.get('name_pattern', '.*')
            if re.match(name_pattern, param_name):
                matched = True
                matched_rule = rule_name
                break

        if matched:
            matched_params.append((param_name, param, matched_rule))
            if verbose:
                print(f"  Matched: {param_name} (shape: {tuple(param.shape)}) [rule: {matched_rule}]")

    return matched_params


# ==================== Hook Functions ====================

def create_fwd_hook_add_perturbation(seed: int, sigma: float, rand_gen_fn: Callable, mask: Optional[torch.Tensor] = None):
    """Create forward hook to add perturbation to activations."""
    def fwd_hook(module, input, output):
        state = torch.get_rng_state()

        if seed is not None:
            torch.manual_seed(seed)
        perturbation = rand_gen_fn(output.shape).to(output.dtype)
        torch.set_rng_state(state)

        if mask is not None:
            perturbation *= mask

        # Store for later use
        module.perturbation = perturbation

        return output + sigma * perturbation
    return fwd_hook


def create_fwd_hook_store_output_shape(module, input, output):
    """Store output shape for later use."""
    module.output_shape = output.shape


def create_bwd_pre_hook_replace_grad(ZO_grad_output: torch.Tensor, debug: bool = False):
    """Create backward pre-hook to replace gradient with ZO estimate."""
    def bwd_pre_hook(module, grad_output):
        if debug:
            import torch.nn.functional as F
            cos_sim = F.cosine_similarity(grad_output[0].reshape(-1), ZO_grad_output.reshape(-1), dim=0)
            print(f'Cosine similarity between true grad and ZO grad: {cos_sim:.4f}')

        if len(grad_output) == 1:
            return (ZO_grad_output,)
        else:
            return (ZO_grad_output,) + grad_output[1:]
    return bwd_pre_hook


# ==================== Spectral Scaling Computation ====================

def compute_layer_dimensions(param: nn.Parameter, layer: Optional[nn.Module] = None) -> Tuple[int, int, int]:
    """
    Compute input/output dimensions and total parameter count for a layer.

    Returns:
        (n_in, n_out, d): input dim, output dim, total params
    """
    shape = param.shape

    if len(shape) == 2:  # Linear layer: (n_out, n_in)
        n_out, n_in = shape
        d = n_in * n_out
    elif len(shape) == 4:  # Conv2d: (n_out, n_in, k, k)
        n_out, n_in, k1, k2 = shape
        d = n_out * n_in * k1 * k2
    elif len(shape) == 1:  # Bias or LayerNorm
        n_out = n_in = shape[0]
        d = n_out
    else:
        # Fallback: treat as flattened
        d = param.numel()
        n_in = n_out = int(d ** 0.5)

    return n_in, n_out, d


def compute_spectral_sigma_wp(n_in: int, n_out: int, d: int, base_sigma: float = 0.01) -> float:
    """
    Compute layer-wise sigma for weight perturbation (WP).

    Formula (from spectral scaling paper):
        σ_WP ∝ 1 / √(n_in + n_out)

    Args:
        n_in: Input dimension
        n_out: Output dimension
        d: Total parameter count (not used in WP, kept for API consistency)
        base_sigma: Base scaling constant (default 0.01)

    Returns:
        Layer-wise sigma
    """
    import math
    return base_sigma * math.sqrt(2.0 / (n_in + n_out))


def compute_spectral_sigma_np(batch_size: int, n_out: int, base_sigma: float = 0.01) -> float:
    """
    Compute layer-wise sigma for node perturbation (NP).

    Formula (from spectral scaling paper):
        σ_NP ∝ 1 / √batch_size

    Args:
        batch_size: Batch size
        n_out: Output activation dimension
        base_sigma: Base scaling constant (default 0.01)

    Returns:
        Layer-wise sigma
    """
    import math
    return base_sigma / math.sqrt(batch_size)


def compute_spectral_lr_mult_wp(n_in: int, n_out: int, d: int,
                                 n_sample: int, sigma: float,
                                 C: float = 1.0, base_lr: float = 1e-3) -> float:
    """
    Compute layer-wise learning rate multiplier for weight perturbation (WP).

    From variance analysis:
        η_ZO = η_FO * √n_sample / (σ * √(C * d))

    From μP scaling (for maintaining spectral norm):
        η_ℓ ∝ n_out / n_in

    Combined:
        lr_mult = (n_out / n_in) * √n_sample / (σ * √(C * d))

    Args:
        n_in: Input dimension
        n_out: Output dimension
        d: Total parameter count
        n_sample: Number of ZO samples
        sigma: Perturbation scale
        C: Curvature constant (default 1.0)
        base_lr: Base learning rate (for reference)

    Returns:
        Learning rate multiplier relative to base_lr
    """
    import math

    # μP scaling component
    mup_scale = n_out / n_in

    # ZO variance adjustment
    zo_adjustment = math.sqrt(n_sample) / (sigma * math.sqrt(C * d))

    return mup_scale * zo_adjustment


def compute_spectral_lr_mult_np(batch_size: int, n_out: int,
                                 n_sample: int, sigma: float,
                                 C: float = 1.0, base_lr: float = 1e-3) -> float:
    """
    Compute layer-wise learning rate multiplier for node perturbation (NP).

    From variance analysis:
        η_ZO = η_FO * √n_sample / (σ * √(C * batch_size * n_out))

    Args:
        batch_size: Batch size
        n_out: Output activation dimension
        n_sample: Number of ZO samples
        sigma: Perturbation scale
        C: Curvature constant (default 1.0)
        base_lr: Base learning rate (for reference)

    Returns:
        Learning rate multiplier relative to base_lr
    """
    import math

    # NP perturbation dimension is batch_size * n_out
    perturb_dim = batch_size * n_out

    # ZO variance adjustment
    zo_adjustment = math.sqrt(n_sample) / (sigma * math.sqrt(C * perturb_dim))

    return zo_adjustment


# ==================== Model Splitting (for partial forward) ====================

def split_model(model: nn.Module, iterable_block_name: Optional[str] = None) -> List[nn.Module]:
    """Split model into modules for partial forward computation."""
    if iterable_block_name is None:
        return [model]

    split_modules = []
    for name, module in model.named_children():
        if iterable_block_name in name:
            for block in module:
                split_modules.append(block)
        else:
            split_modules.append(module)

    return split_modules
