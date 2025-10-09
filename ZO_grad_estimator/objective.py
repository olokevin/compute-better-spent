"""
Objective functions for different tasks.

Each objective function wraps the forward pass and loss computation,
providing a consistent interface for ZO gradient estimation.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple, Any


class ObjectiveFunction:
    """Base class for objective functions."""

    def __call__(self, return_loss_reduction: str = 'mean', **kwargs) -> Tuple[Any, torch.Tensor]:
        """
        Execute forward pass and loss computation.

        Args:
            return_loss_reduction: 'mean', 'none', or 'sum'
            **kwargs: Additional arguments

        Returns:
            (output, loss) tuple
        """
        raise NotImplementedError


class CIFARObjective(ObjectiveFunction):
    """Objective function for CIFAR classification."""

    def __init__(self, model: nn.Module, ims: torch.Tensor, targs: torch.Tensor,
                 loss_fn: nn.Module, args: Any):
        self.model = model
        self.ims = ims
        self.targs = targs
        self.loss_fn = loss_fn
        self.args = args

    def _compute_loss(self, preds: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute loss with mixup support."""
        if self.args.mixup > 0:
            targs_perm = self.targs[:, 1].long()
            weight = self.targs[0, 2].squeeze()
            targs = self.targs[:, 0].long()

            if weight != -1:
                if reduction == 'none':
                    loss1 = nn.functional.cross_entropy(preds, targs, reduction='none')
                    loss2 = nn.functional.cross_entropy(preds, targs_perm, reduction='none')
                    loss = loss1 * weight + loss2 * (1 - weight)
                else:
                    loss1 = self.loss_fn(preds, targs)
                    loss2 = self.loss_fn(preds, targs_perm)
                    loss = loss1 * weight + loss2 * (1 - weight)
            else:
                if reduction == 'none':
                    loss = nn.functional.cross_entropy(preds, targs, reduction='none')
                else:
                    loss = self.loss_fn(preds, targs)
        else:
            targs = self.targs

            if self.args.ar_modeling:
                targs = rearrange(self.ims, 'b c h w -> (b h w c)')
                preds = preds[:, :-1].reshape(-1, preds.shape[-1])

            if reduction == 'none':
                loss = nn.functional.cross_entropy(preds, targs, reduction='none')
            else:
                loss = self.loss_fn(preds, targs)

        return loss

    def __call__(self, return_loss_reduction: str = 'mean', **kwargs) -> Tuple[Any, torch.Tensor]:
        # Forward pass
        preds = self.model(self.ims)

        # Compute base loss
        loss = self._compute_loss(preds, reduction=return_loss_reduction)

        # Add auxiliary losses (only for mean reduction)
        if return_loss_reduction == 'mean':
            aux_losses = []
            spec_penalties = []

            for name, module in self.model.named_modules():
                if 'moe_gate' in name:
                    aux_losses.append(module.load_balancing_loss)
                if hasattr(module, 'natural_norm'):
                    spec_penalties.append(module.natural_norm**2)

            aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else 0
            spec_penalty = sum(spec_penalties) / len(spec_penalties) if spec_penalties else 0

            aux_loss_weight = getattr(self.args, 'aux_loss_weight', 0.01)
            spec_penalty_weight = getattr(self.args, 'spec_penalty_weight', 0.0)

            loss = loss + aux_loss * aux_loss_weight + spec_penalty * spec_penalty_weight

        # Wrap output
        class Output:
            def __init__(self, logits):
                self.logits = logits

        output = Output(preds)
        return output, loss


class LMObjective(ObjectiveFunction):
    """Objective function for language modeling."""

    def __init__(self, model: nn.Module, batch: dict):
        self.model = model
        self.batch = batch

    def _get_per_sample_loss(self, output: Any) -> torch.Tensor:
        """Compute per-sample loss for language modeling."""
        logits = output.logits.float()
        labels = self.batch['labels']

        batch_sz, seq_len = logits.size(0), logits.size(1)
        vocab_size = logits.size(-1)

        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(logits.device)

        loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction='none')
        loss = loss.reshape(batch_sz, seq_len)

        return loss

    def __call__(self, return_loss_reduction: str = 'mean', **kwargs) -> Tuple[Any, torch.Tensor]:
        output = self.model(**self.batch)

        if return_loss_reduction == 'mean':
            loss = output.loss
        elif return_loss_reduction == 'none':
            loss = self._get_per_sample_loss(output)
        else:
            raise ValueError(f"Unknown reduction: {return_loss_reduction}")

        return output, loss


def build_objective_function(obj_fn_type: str, **kwargs) -> ObjectiveFunction:
    """Build objective function based on task type."""
    if obj_fn_type == 'CIFAR':
        return CIFARObjective(**kwargs)
    elif obj_fn_type == 'LM':
        return LMObjective(**kwargs)
    else:
        raise NotImplementedError(f"Unknown obj_fn_type: {obj_fn_type}")
