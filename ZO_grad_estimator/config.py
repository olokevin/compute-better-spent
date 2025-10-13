"""
Configuration system for ZO gradient estimator.

Uses YAML files to specify:
- Perturbation strategy (weight vs node)
- Layer/parameter selection rules (regex-based)
- Sampling methods and hyperparameters
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class ZOConfig:
    """Configuration for ZO gradient estimation."""

    # === Core Settings ===
    name: str = 'ZO_Estim_MC'  # Estimator type
    obj_fn_type: str = 'CIFAR'  # Objective function type: 'CIFAR', 'LM', etc.

    # === Perturbation Parameters ===
    sigma: float = 0.01  # Perturbation magnitude
    n_sample: int = 1  # Number of random samples per gradient estimate
    estimate_method: str = 'antithetic'  # 'forward' or 'antithetic'
    sample_method: str = 'bernoulli'  # 'gaussian', 'bernoulli', or 'uniform'

    # === Optimization Options ===
    signsgd: bool = False  # Use sign of gradient only
    quantized: bool = False  # Use quantized perturbations
    normalize_perturbation: bool = False  # Normalize perturbation to unit norm
    scale: Optional[str] = None  # Gradient scaling: None, 'sqrt_dim', or 'dim'

    # === Spectral Scaling Options ===
    en_spectral_scaling: bool = False  # Enable layer-wise spectral scaling for sigma and LR
    spectral_sigma_method: str = 'wp_standard'  # 'wp_standard', 'np_standard', or 'custom'
    spectral_lr_method: str = 'zo_variance_adjusted'  # 'zo_variance_adjusted', 'fo_baseline', or 'custom'
    spectral_C_constant: float = 1.0  # Curvature constant C for variance estimation

    # === Strategy Flags ===
    en_layerwise_perturbation: bool = False  # Perturb one layer at a time
    en_partial_forward: bool = False  # Use partial forward (requires model support)
    en_wp_np_mixture: bool = False  # Mix weight and node perturbation
    en_pseudo_ZO: bool = False  # Exclude certain layers (e.g., classifier)
    en_param_commit: bool = False  # Commit param changes during perturbation

    # === Rule-Based Selection ===
    # Parameter perturbation rules (for weight perturbation)
    param_perturb_rules: Optional[Dict[str, Dict[str, Any]]] = None

    # Activation perturbation rules (for node perturbation)
    actv_perturb_rules: Optional[Dict[str, Dict[str, Any]]] = None

    # Legacy support (deprecated, use rules instead)
    param_perturb_block_idx_list: Optional[List[int]] = None
    actv_perturb_block_idx_list: Optional[Any] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ZOConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'obj_fn_type': self.obj_fn_type,
            'sigma': self.sigma,
            'n_sample': self.n_sample,
            'estimate_method': self.estimate_method,
            'sample_method': self.sample_method,
            'signsgd': self.signsgd,
            'quantized': self.quantized,
            'normalize_perturbation': self.normalize_perturbation,
            'scale': self.scale,
            'en_spectral_scaling': self.en_spectral_scaling,
            'spectral_sigma_method': self.spectral_sigma_method,
            'spectral_lr_method': self.spectral_lr_method,
            'spectral_C_constant': self.spectral_C_constant,
            'en_layerwise_perturbation': self.en_layerwise_perturbation,
            'en_partial_forward': self.en_partial_forward,
            'en_wp_np_mixture': self.en_wp_np_mixture,
            'en_pseudo_ZO': self.en_pseudo_ZO,
            'en_param_commit': self.en_param_commit,
            'param_perturb_rules': self.param_perturb_rules,
            'actv_perturb_rules': self.actv_perturb_rules,
            'param_perturb_block_idx_list': self.param_perturb_block_idx_list,
            'actv_perturb_block_idx_list': self.actv_perturb_block_idx_list,
        }
