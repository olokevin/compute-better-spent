"""
Test script for gradient scaling in ZO estimator.

Tests:
1. Weight perturbation with different scale settings
2. Node perturbation with different scale settings
"""

import torch
import torch.nn as nn
import math
from .config import ZOConfig
from .estimator import ZOEstimator
from .objective import CIFARObjective


class SimpleModel(nn.Module):
    """Simple 2-layer MLP for testing."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_weight_perturbation_scaling():
    """Test gradient scaling for weight perturbation."""
    print("\n=== Testing Weight Perturbation Scaling ===\n")

    # Create model
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create dummy data
    batch_size = 8
    ims = torch.randn(batch_size, 10).to(device)
    targs = torch.randint(0, 5, (batch_size,)).to(device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Create a simple args object
    class Args:
        mixup = 0
        ar_modeling = False
        aux_loss_weight = 0.01
        spec_penalty_weight = 0.0

    args = Args()

    # Test different scale settings
    scale_options = [None, 'sqrt_dim', 'dim']

    for scale in scale_options:
        print(f"\n--- Testing scale={scale} ---")

        # Create config
        config = ZOConfig(
            sigma=0.01,
            n_sample=1,
            estimate_method='forward',
            sample_method='gaussian',
            scale=scale,
            param_perturb_rules={
                'all_fc': {
                    'name_pattern': 'fc.*',
                    'type_filter': 'Linear'
                }
            }
        )

        # Create estimator
        zo_estimator = ZOEstimator(config, model)

        # Create objective
        obj_fn = CIFARObjective(
            model=model,
            ims=ims,
            targs=targs,
            loss_fn=loss_fn,
            args=args
        )

        zo_estimator.update_objective(obj_fn)

        # Estimate gradients
        model.eval()
        output, loss = zo_estimator.estimate_grad()

        # Check gradients exist
        grad_found = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_found = True
                grad_norm = param.grad.norm().item()
                param_dim = param.numel()

                # Compute expected scaling factor
                if scale is None:
                    expected_scale_mult = 1.0
                elif scale == 'sqrt_dim':
                    expected_scale_mult = math.sqrt(config.n_sample / (config.n_sample + param_dim - 1))
                elif scale == 'dim':
                    expected_scale_mult = (config.n_sample / (config.n_sample + param_dim - 1))

                print(f"  {name}: grad_norm={grad_norm:.6f}, param_dim={param_dim}, "
                      f"expected_scale_mult={expected_scale_mult:.6f}")

        if grad_found:
            print(f"  ✓ Gradients computed successfully")
        else:
            print(f"  ✗ No gradients found!")

        # Clean up gradients
        model.zero_grad()


def test_node_perturbation_scaling():
    """Test gradient scaling for node perturbation."""
    print("\n=== Testing Node Perturbation Scaling ===\n")

    # Create model
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create dummy data
    batch_size = 8
    ims = torch.randn(batch_size, 10).to(device)
    targs = torch.randint(0, 5, (batch_size,)).to(device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Create a simple args object
    class Args:
        mixup = 0
        ar_modeling = False
        aux_loss_weight = 0.01
        spec_penalty_weight = 0.0

    args = Args()

    # Test different scale settings
    scale_options = [None, 'sqrt_dim', 'dim']

    for scale in scale_options:
        print(f"\n--- Testing scale={scale} ---")

        # Create config
        config = ZOConfig(
            sigma=0.01,
            n_sample=1,
            estimate_method='forward',
            sample_method='gaussian',
            scale=scale,
            actv_perturb_rules={
                'fc1': {
                    'name_pattern': 'fc1',
                    'type_filter': 'Linear'
                }
            }
        )

        # Create estimator
        zo_estimator = ZOEstimator(config, model)

        # Create objective
        obj_fn = CIFARObjective(
            model=model,
            ims=ims,
            targs=targs,
            loss_fn=loss_fn,
            args=args
        )

        zo_estimator.update_objective(obj_fn)

        # Estimate gradients
        model.eval()
        zo_estimator.estimate_grad()

        # Check ZO_grad_output exists
        zo_grad_found = False
        for perturb_layer in zo_estimator.perturb_layers:
            if hasattr(perturb_layer.layer, 'ZO_grad_output'):
                zo_grad_found = True
                zo_grad_norm = perturb_layer.layer.ZO_grad_output.norm().item()
                output_shape = perturb_layer.layer.output_shape
                actv_dim = torch.tensor(output_shape).prod().item() / batch_size

                # Compute expected scaling factor
                if scale is None:
                    expected_scale_mult = 1.0
                elif scale == 'sqrt_dim':
                    expected_scale_mult = math.sqrt(batch_size * config.n_sample /
                                                    (batch_size * config.n_sample + actv_dim - 1))
                elif scale == 'dim':
                    expected_scale_mult = (batch_size * config.n_sample /
                                          (batch_size * config.n_sample + actv_dim - 1))

                print(f"  {perturb_layer.name}: ZO_grad_norm={zo_grad_norm:.6f}, "
                      f"actv_dim={actv_dim:.0f}, expected_scale_mult={expected_scale_mult:.6f}")

        if zo_grad_found:
            print(f"  ✓ ZO gradients computed successfully")
        else:
            print(f"  ✗ No ZO gradients found!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Gradient Scaling Implementation")
    print("=" * 60)

    # Test weight perturbation
    test_weight_perturbation_scaling()

    # Test node perturbation
    test_node_perturbation_scaling()

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
