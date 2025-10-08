from typing import Callable

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from easydict import EasyDict

from scipy.stats import qmc
from .ZO_utils import split_model, split_named_model, SplitedLayer, SplitedParam, default_create_fwd_hook_add_perturbation, default_create_fwd_hook_get_out_dimension
from .ZO_utils import build_rand_gen_fn
from .ZO_utils import default_wp_add_perturbation, default_wp_remove_perturbation, default_wp_gen_grad
# from .QMC_sampler import sphere_n, coord_basis, block_mask_generator, layer_mask_generator

DEBUG = False
# DEBUG = True

class ZO_Estim_Base(nn.Module):
    """
    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
    """
    
    def __init__(
        self,
        model: nn.Module,
        obj_fn_type: str,
        
        # For param perturb ZO. A list of SplitedParam. Specifies what Tensors should be optimized.
        splited_param_list: list = None,
        # For actv  perturb ZO. A list of SplitedLayer. Specifies what layers' activations should be perturbed.
        splited_layer_list: list = None,
        
        config = EasyDict({
            "sigma": 0.1,
            "n_sample": 20,
            "signsgd": False,
            "quantized": False,
            "estimate_method": 'forward',
            "sample_method": 'gaussian',
            "normalize_perturbation": False,
            "scale": None,
            "en_layerwise_perturbation": True,
            "en_partial_forward": True,
            "en_param_commit": False,
          }),
        ):
        super().__init__()

        self.model = model
        self.obj_fn_type = obj_fn_type

        self.splited_param_list = splited_param_list
        self.splited_layer_list = splited_layer_list
        
        self.splited_modules_list = split_model(model)
    
        self.sigma = config.sigma
        self.n_sample = config.n_sample
        self.signsgd = config.signsgd

        self.quantized = config.quantized
        self.estimate_method = config.estimate_method
        self.sample_method = config.sample_method
        self.normalize_perturbation = config.normalize_perturbation
        
        self.scale =config.scale
        
        self.en_layerwise_perturbation = config.en_layerwise_perturbation
        self.en_partial_forward = config.en_partial_forward
        self.en_param_commit = config.en_param_commit
        
        self.config = config

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        self.rand_gen_fn = build_rand_gen_fn(self.sample_method, self.device)
        
        self.ZO_dimension = None
        
        self.forward_counter = 0
            
    ### Generate random vectors from a normal distribution
    def _generate_random_vector(self, shape, sample_method, device):
        
        dimension = np.prod(shape)

        if self.quantized == True:
            if sample_method == 'bernoulli':
                sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
            else:
                return NotImplementedError('Unlnown sample method', self.sample_method)
        else:
            if sample_method == 'uniform':
                sample = torch.randn(shape, device=device)
                sample = torch.nn.functional.normalize(sample, p=2, dim=0)
            elif sample_method == 'gaussian':
                sample = torch.randn(shape, device=device)
                # sample = torch.randn(shape, device=device) / dimension
            elif sample_method == 'bernoulli':
                ### Rademacher
                sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
            elif sample_method in ('sobol', 'halton'):
                if self.sampler == None:
                    raise ValueError('Need sampler input')
                else:
                    sample = torch.Tensor(self.sampler.random(1)).squeeze()
                    sample = 2*sample-torch.ones_like(sample)
                    sample = torch.nn.functional.normalize(sample, p=2, dim=0)
                    sample = sample.to(device)
            elif sample_method == 'sphere_n':
                sample = next(self.sampler)
                sample = sample.to(device)
            else:
                return NotImplementedError('Unlnown sample method', sample_method)
            
        return sample
    
    def update_obj_fn(self, obj_fn):
        self.obj_fn = obj_fn
        
        if self.ZO_dimension is None:
            self.ZO_dimension = self.get_ZO_dimension()
    
    def get_forward_cnt(self):
        return self.forward_counter
    
    def get_ZO_dimension(self):
        ZO_dimension = 0
        if self.splited_param_list is not None:
            for splited_param in self.splited_param_list:
                print(f'{splited_param.name} param_dimension={splited_param.param.numel()}')
                ZO_dimension += splited_param.param.numel()
                # param.grad = torch.zeros_like(param)

        if self.splited_layer_list is not None:
            ### get activation dimension
            fwd_hook_handle_list = []
            for splited_layer in self.splited_layer_list:
                create_fwd_hook_get_out_dimension = getattr(splited_layer.layer, 'create_fwd_hook_get_out_dimension', default_create_fwd_hook_get_out_dimension)
                fwd_hook_get_out_dimension = create_fwd_hook_get_out_dimension()
                fwd_hook_handle_list.append(splited_layer.layer.register_forward_hook(fwd_hook_get_out_dimension))
                    
            _, old_loss = self.obj_fn(return_loss_reduction='none')
            
            for splited_layer in self.splited_layer_list:
                if splited_layer.mode == 'actv':
                    print(f'{splited_layer.name} out_dimension={splited_layer.layer.out_dimension}')
                    ZO_dimension += splited_layer.layer.out_dimension

                elif splited_layer.mode == 'param':
                    param_dim = 0
                    for param in splited_layer.layer.parameters():
                        if param.requires_grad:
                            param_dim += param.numel()
                    print(f'{splited_layer.name} param_dimension={param_dim}')
                
                    ZO_dimension += param_dim
                    
            for fwd_hook_handle in fwd_hook_handle_list:
                fwd_hook_handle.remove()  
        
        ZO_dimension = int(ZO_dimension)
        print('ZO_dimension=', ZO_dimension)
        return ZO_dimension