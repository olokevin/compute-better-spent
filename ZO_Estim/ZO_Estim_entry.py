"""
ZO Estimator Entry Point with Regex-Based Layer Selection

This module provides:
1. Flexible rule-based system for selecting layers/parameters for perturbation
2. Building ZO gradient estimators (MC and PZO modes)
3. Objective functions for different tasks (LM, CIFAR)
"""

import re
import torch
import torch.nn as nn
from einops import rearrange

from .ZO_utils import SplitedLayer, SplitedParam, split_model
from .ZO_Estim_MC import ZO_Estim_MC
from nn.cola_nn import CoLALayer as ColaLayer

def get_type_mapping():
    """
    Map string type names to actual class types.

    This allows YAML configs to specify types as strings like 'CoLALayer' or 'Linear',
    which get resolved to actual Python classes at runtime.

    Returns:
        dict: Mapping from string names to class types
    """
    type_map = {
        'CoLALayer': ColaLayer,
        'nn.Linear': nn.Linear,
        'nn.Conv2d': nn.Conv2d,
        'nn.LayerNorm': nn.LayerNorm,
        'nn.BatchNorm2d': nn.BatchNorm2d,
    }

    # Add more types as needed
    return type_map


def resolve_type_strings(rule_spec):
    """
    Convert string type names in rules to actual class types.

    Args:
        rule_spec (dict): Rule specification that may contain 'type' field with strings

    Returns:
        dict: Rule spec with resolved type classes
    """
    if 'type' not in rule_spec:
        return rule_spec

    type_map = get_type_mapping()
    types = rule_spec['type']

    # Handle single string or list of strings
    if isinstance(types, str):
        types = [types]

    # Resolve string names to actual classes
    resolved_types = []
    for t in types:
        if isinstance(t, str):
            if t in type_map:
                resolved_types.append(type_map[t])
            else:
                print(f"Warning: Unknown type '{t}', skipping")
        else:
            # Already a class type
            resolved_types.append(t)

    # Update rule_spec with resolved types
    resolved_spec = rule_spec.copy()
    resolved_spec['type'] = resolved_types
    return resolved_spec


def match_layer_by_rules(layer_name, layer, rules):
    """
    Check if a layer matches any of the selection rules.

    Args:
        layer_name (str): Full name of the layer
        layer (nn.Module): The layer module
        rules (dict): Dictionary of rules with name_pattern and optional type
                     Type can be specified as string (e.g., 'CoLALayer') or class

    Returns:
        tuple: (matched, rule_name)
    """
    for rule_name, rule_spec in rules.items():
        # Resolve string type names to classes
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
            # Handle both type objects and instances
            if not any(isinstance(layer, t) if isinstance(t, type) else type(layer) == t for t in allowed_types):
                continue

        return True, rule_name

    return False, None


def find_layers_by_rules(model, rules, require_grad=True, verbose=True):
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


def find_params_by_rules(model, rules, require_grad=True, verbose=True):
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


def build_ZO_Estim(config, model):
    """
    Build ZO gradient estimator with flexible rule-based layer selection.

    Config should contain either:
    - param_perturb_rules: dict of rules for parameter perturbation
    - actv_perturb_rules: dict of rules for activation perturbation

    Old-style configs with param_perturb_block_idx_list/actv_perturb_block_idx_list
    are still supported for backward compatibility.
    """
    if config.name == 'ZO_Estim_MC':
        ### split model
        ZO_iterable_block_name = getattr(model, 'ZO_iterable_block_name', None)
        split_modules_list = split_model(model, ZO_iterable_block_name)

        splited_param_list = None
        splited_layer_list = None

        ### ===== PARAMETER PERTURBATION =====
        # New rule-based system
        if hasattr(config, 'param_perturb_rules') and config.param_perturb_rules is not None:
            print('\n=== Parameter Perturbation (Rule-Based) ===')
            matched_params = find_params_by_rules(
                model,
                config.param_perturb_rules,
                require_grad=True,
                verbose=True
            )

            splited_param_list = []
            for param_name, param, rule_name in matched_params:
                splited_param_list.append(
                    SplitedParam(idx=-1, name=param_name, layer=None, param=param)
                )

        # Legacy support
        elif hasattr(config, 'param_perturb_block_idx_list') and config.param_perturb_block_idx_list is not None:
            print('\n=== Parameter Perturbation (Legacy Mode) ===')
            splited_param_list = []
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"  {param_name}")
                    splited_param_list.append(
                        SplitedParam(idx=-1, name=param_name, layer=None, param=param)
                    )

        ### ===== ACTIVATION PERTURBATION =====
        # New rule-based system
        if hasattr(config, 'actv_perturb_rules') and config.actv_perturb_rules is not None:
            print('\n=== Activation Perturbation (Rule-Based) ===')
            matched_layers = find_layers_by_rules(
                model,
                config.actv_perturb_rules,
                require_grad=True,
                verbose=True
            )

            splited_layer_list = []
            for layer_name, layer, rule_name in matched_layers:
                splited_layer_list.append(
                    SplitedLayer(idx=-1, name=layer_name, layer=layer)
                )

        # Legacy support
        elif hasattr(config, 'actv_perturb_block_idx_list') and config.actv_perturb_block_idx_list is not None:
            print('\n=== Activation Perturbation (Legacy Mode) ===')
            splited_layer_list = []

            for layer_name, layer in model.named_modules():
                if type(layer) in (ColaLayer,):
                    if all(param.requires_grad for param in layer.parameters()):
                        print(f"  {layer_name}")
                        splited_layer_list.append(
                            SplitedLayer(idx=-1, name=layer_name, layer=layer)
                        )


        ### Handle pseudo-ZO exclusions
        if splited_layer_list is not None:
            if getattr(config, 'en_pseudo_ZO', False):
                original_count = len(splited_layer_list)
                splited_layer_list = [
                    layer for layer in splited_layer_list
                    if 'classifier' not in layer.name
                ]
                print(f'\nPseudo-ZO: Excluded {original_count - len(splited_layer_list)} classifier layers')

        ### Mixed weight/node perturbation (advanced)
        ZO_trainable_layers_list_wp = None
        if getattr(config, 'en_wp_np_mixture', False):
            if hasattr(model, 'ZO_trainable_layers_list_wp'):
                ZO_trainable_layers_list_wp = model.ZO_trainable_layers_list_wp

        if splited_layer_list is not None and ZO_trainable_layers_list_wp is not None:
            for splited_layer in splited_layer_list:
                if isinstance(splited_layer.layer, ZO_trainable_layers_list_wp):
                    splited_layer.mode = 'param'

        ### Build ZO estimator
        print(f'\n=== ZO Estimator Summary ===')
        print(f'Parameters to perturb: {len(splited_param_list) if splited_param_list else 0}')
        print(f'Layers to perturb: {len(splited_layer_list) if splited_layer_list else 0}')

        ZO_Estim = ZO_Estim_MC(
            model=model,
            obj_fn_type=config.obj_fn_type,
            splited_param_list=splited_param_list,
            splited_layer_list=splited_layer_list,
            config=config,
        )
        return ZO_Estim
    elif config.name == 'ZO_Estim_PZO':
        from cola.modeling_cola import ColaDecoderLayer
        from .ZO_Estim_PZO import PZOSplitedLayer, ZO_Estim_PZO
        from collections import OrderedDict

        if config.actv_perturb_block_idx_list == 'all':
            actv_perturb_block_idx_list = list(range(len(model.model.layers)))
        else:
            actv_perturb_block_idx_list = config.actv_perturb_block_idx_list

        trainable_layer_dict = OrderedDict()

        print('PZO trainable layers')
        for layer_name, layer in model.named_modules():
            ### trainable layer type
            if type(layer) in (ColaDecoderLayer,):
                ### trainable encoder idx
                if any([str(x) in layer_name for x in actv_perturb_block_idx_list]):
                    ### trainable layer
                    if all(param.requires_grad for param in layer.parameters()):
                        trainable_layer_dict[layer_name] = layer
                        print('layer', layer_name)

        splited_layer_list = []
        layer_names = list(trainable_layer_dict.keys())

        for idx, layer_name in enumerate(layer_names):
            current_layer = trainable_layer_dict[layer_name]
            # For the last layer, perturb_layer will be None
            perturb_layer = trainable_layer_dict[layer_names[idx + 1]].input_layernorm if idx < len(layer_names) - 1 else None
            splited_layer_list.append(PZOSplitedLayer(idx=idx, name=layer_name, layer=current_layer, perturb_layer=perturb_layer))


        ZO_Estim = ZO_Estim_PZO(
            model = model,
            obj_fn_type = config.obj_fn_type,
            splited_layer_list = splited_layer_list,
            config = config,
        )
        return ZO_Estim
    else:
        raise NotImplementedError(f"Unknown ZO estimator type: {config.name}")


def build_obj_fn(obj_fn_type, **kwargs):
    """Build objective function based on task type."""
    if obj_fn_type == 'LM':
        obj_fn = ObjFnLM(**kwargs)
    elif obj_fn_type == 'LM_PZO':
        obj_fn = PZO_ObjFnLM(**kwargs)
    elif obj_fn_type == 'CIFAR':
        obj_fn = ObjFnCIFAR(**kwargs)
    else:
        raise NotImplementedError(f"Unknown obj_fn_type: {obj_fn_type}")
    return obj_fn


### ===== OBJECTIVE FUNCTIONS ===== ###

class ObjFnLM:
    """Objective function for language modeling tasks."""

    def __init__(self, model, batch):
        self.model = model
        self.batch = batch

    def get_none_reduction_loss(self, output):
        logits = output.logits.float()
        labels = self.batch['labels']

        batch_sz = logits.size(0)
        seq_len = logits.size(1)
        vocab_size = logits.size(-1)

        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(logits.device)
        loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction='none').reshape(batch_sz, seq_len)
        # print('loss shape', loss.shape)
        return loss

    def get_one_token_loss(self, output):
        logits = output.logits.float()
        labels = self.batch['labels']

        batch_sz = logits.size(0)
        seq_len = logits.size(1)
        vocab_size = logits.size(-1)

        # original logits: [batch_sz, seq_len, vocab_size]
        # original labels: [batch_sz, seq_len]

        # Shift labels as before:
        labels_padded   = nn.functional.pad(self.batch['labels'], (0, 1), value=-100)  # [batch_sz, seq_len+1]
        shift_labels    = labels_padded[..., 1:]                                      # [batch_sz, seq_len]

        # pick token i
        i = 100
        logits_i = output.logits[:, i, :].float()     # [batch_sz, vocab_size]
        labels_i = shift_labels[:, i].to(logits_i.device)  # [batch_sz]

        # compute only that token's loss
        loss = torch.nn.functional.cross_entropy(logits_i, labels_i,
                                ignore_index=-100,
                                reduction='mean')      # [batch_sz]
        # or reduction='mean' to get a scalar
        return loss

    def __call__(self, return_loss_reduction='mean'):
        output = self.model(**self.batch)

        if return_loss_reduction == 'mean':
            loss = output.loss
        elif return_loss_reduction == 'none':
            loss = self.get_none_reduction_loss(output)
        elif return_loss_reduction == 'one-token':
            loss = self.get_one_token_loss(output)

        return output, loss


class PZO_ObjFnLM(ObjFnLM):
    """Objective function for PZO language modeling."""

    def __init__(self, model, batch):
        super().__init__(model, batch)

    def get_grad_hidden_states(self, hidden_states):
        hidden_states.requires_grad = True
        logits = self.model.module.lm_head(hidden_states)
        labels = self.batch['labels']

        batch_sz = logits.size(0)
        seq_len = logits.size(1)
        vocab_size = logits.size(-1)

        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(logits.device)
        loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction='mean')

        # Compute gradients with respect to hidden_states
        grad_hidden_states = torch.autograd.grad(loss, hidden_states, create_graph=False)[0]

        return grad_hidden_states

    def get_hidden_states(self):
        batch = self.batch.copy()
        batch['output_hidden_states'] = True

        output = self.model(**batch)

        return output.hidden_states[0]

    def __call__(self, return_loss_reduction='mean'):
        output = self.model(**self.batch)

        if return_loss_reduction == 'mean':
            loss = output.loss
        elif return_loss_reduction == 'none':
            loss = self.get_none_reduction_loss(output)

        return output, loss


class ObjFnCIFAR:
    """
    Objective function wrapper for CIFAR classification.

    This class encapsulates the forward pass and loss computation for CIFAR training,
    providing a consistent interface for ZO gradient estimation.

    Args:
        model (nn.Module): The neural network model
        ims (torch.Tensor): Input images, shape (batch_size, channels, height, width)
        targs (torch.Tensor): Target labels, shape (batch_size,) or (batch_size, 3) for mixup
        loss_fn (callable): Loss function (e.g., CrossEntropyLoss)
        args (Namespace): Training arguments containing:
            - mixup (float): Mixup strength (0 to disable)
            - ar_modeling (bool): Whether using autoregressive modeling

    Example:
        >>> obj_fn = ObjFnCIFAR(model, ims, targs, loss_fn, args)
        >>> output, loss = obj_fn()  # Standard call with mean reduction
        >>> output, loss = obj_fn(return_loss_reduction='none')  # Per-sample losses
    """

    def __init__(self, model, ims, targs, loss_fn, args):
        self.model = model
        self.ims = ims
        self.targs = targs
        self.loss_fn = loss_fn
        self.args = args

    def _compute_loss(self, preds, reduction='mean'):
        """
        Compute loss with optional mixup support.

        Args:
            preds (torch.Tensor): Model predictions
            reduction (str): Loss reduction mode ('mean', 'none', or 'sum')

        Returns:
            torch.Tensor: Computed loss
        """
        # Handle mixup augmentation
        if self.args.mixup > 0:
            # Mixup format: targs[:, 0] = label1, targs[:, 1] = label2, targs[0, 2] = weight
            targs_perm = self.targs[:, 1].long()
            weight = self.targs[0, 2].squeeze()
            targs = self.targs[:, 0].long()

            if weight != -1:
                # Compute mixup loss
                if reduction == 'none':
                    # For per-sample losses, we need to handle mixup differently
                    loss1 = nn.functional.cross_entropy(preds, targs, reduction='none')
                    loss2 = nn.functional.cross_entropy(preds, targs_perm, reduction='none')
                    loss = loss1 * weight + loss2 * (1 - weight)
                else:
                    loss1 = self.loss_fn(preds, targs)
                    loss2 = self.loss_fn(preds, targs_perm)
                    loss = loss1 * weight + loss2 * (1 - weight)
            else:
                # No mixup for this batch
                if reduction == 'none':
                    loss = nn.functional.cross_entropy(preds, targs, reduction='none')
                else:
                    loss = self.loss_fn(preds, targs)
        else:
            # Standard classification
            targs = self.targs

            # Handle autoregressive modeling (treating images as sequences)
            if self.args.ar_modeling:
                targs = rearrange(self.ims, 'b c h w -> (b h w c)')
                preds = preds[:, :-1].reshape(-1, preds.shape[-1])

            # Compute loss
            if reduction == 'none':
                loss = nn.functional.cross_entropy(preds, targs, reduction='none')
            else:
                loss = self.loss_fn(preds, targs)

        return loss

    def __call__(self, return_loss_reduction='mean', **kwargs):
        """
        Execute forward pass and loss computation.

        Args:
            return_loss_reduction (str): How to reduce the loss
                - 'mean': Return mean loss (scalar)
                - 'none': Return per-sample losses (for ZO estimation)
                - 'sum': Return sum of losses
            **kwargs: Additional arguments (for compatibility with other objective functions)

        Returns:
            tuple: (output, loss) where:
                - output: Object with .logits attribute containing model predictions
                - loss: Computed loss (scalar or per-sample depending on reduction)
        """
        # Forward pass
        preds = self.model(self.ims)

        # Compute base loss
        loss = self._compute_loss(preds, reduction=return_loss_reduction)

        # Add auxiliary losses (MoE load balancing, spectral penalties, etc.)
        # Only add these for mean reduction (not for per-sample ZO estimation)
        if return_loss_reduction == 'mean':
            aux_losses = []
            spec_penalties = []

            for name, module in self.model.named_modules():
                # MoE load balancing loss
                if 'moe_gate' in name:
                    aux_losses.append(module.load_balancing_loss)
                # Spectral penalty (natural norm regularization)
                if hasattr(module, 'natural_norm'):
                    spec_penalties.append(module.natural_norm**2)

            aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else 0
            spec_penalty = sum(spec_penalties) / len(spec_penalties) if spec_penalties else 0

            # Apply weights (default: aux_loss_weight=0.01, spec_penalty_weight=0.0)
            aux_loss_weight = getattr(self.args, 'aux_loss_weight', 0.01)
            spec_penalty_weight = getattr(self.args, 'spec_penalty_weight', 0.0)

            loss = loss + aux_loss * aux_loss_weight + spec_penalty * spec_penalty_weight

        # Wrap output to match expected interface
        # (ZO_Estim expects output.logits for some methods)
        class Output:
            def __init__(self, logits):
                self.logits = logits

        output = Output(preds)

        return output, loss
