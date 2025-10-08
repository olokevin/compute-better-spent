from typing import Callable

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .ZO_utils import SplitedLayer, default_create_bwd_pre_hook_ZO_grad, fwd_pre_hook_detach_input, fwd_hook_detach_output
from .ZO_Estim_Base import ZO_Estim_Base

DEBUG = False
# DEBUG = True

class PZOSplitedLayer(SplitedLayer):
    def __init__(self, perturb_layer, **kwargs):
        super().__init__(**kwargs)
        self.perturb_layer = perturb_layer
        self.create_fwd_hook_add_perturbation = default_create_fwd_hook_add_perturbation
        self.create_fwd_pre_hook_add_perturbation = default_create_fwd_pre_hook_add_perturbation

def default_create_fwd_hook_add_perturbation(seed, sigma, rand_gen_fn, mask=None, token_idx=None):
    def fwd_hook(module, input, output):
        # input is a tuple

        # output is a tensor. inplace & return modifiled output both work
        if type(output) == tuple:
            p_output = output[0]
        else:
            p_output = output
        
        # state = torch.get_rng_state()
        
        if seed is not None:
            torch.manual_seed(seed)
            
        ### independent perturbation for all tokens
        u = rand_gen_fn(p_output.shape).to(p_output.dtype)
        
        ### shared perturbation within a sequence
        # u = rand_gen_fn((p_output.size(0), p_output.size(-1))).to(p_output.dtype)
        # u = u.unsqueeze(1).expand(-1, p_output.size(1), -1)
        
        ### LRT
        # u = 
        
        ### power law scaling, sigma = C j^{-1/2}, u = sigma * e
        # scale = torch.arange(1, u.size(1)+1).pow(-0.1).to(p_output.dtype).to(p_output.device)
        # u *= scale.reshape(1, -1, 1)
        
        # torch.set_rng_state(state)
        
        if token_idx is not None: 
            u_mask = torch.zeros_like(u)
            u_mask[:, token_idx, :] = 1
            u = u * u_mask
        
        if mask is not None:
            u = u * mask
        
        # print(f'u tokenwise norm {u.norm(dim=(0,2))}')
            
        module.perturbation = u
        
        if type(output) == tuple:
          return (p_output + sigma * u, output[1:])
        else: 
            return p_output + sigma * u
    return fwd_hook
  
def default_create_fwd_pre_hook_add_perturbation(seed, sigma, rand_gen_fn, mask=None, token_idx=None):
    def fwd_hook(module, input, output):
        # input is a tuple
        # module.in_value = input[0].detach().clone()
        # output is a tensor. inplace & return modifiled output both work
        
        # state = torch.get_rng_state()
        
        if seed is not None:
            torch.manual_seed(seed)
            
        ### independent perturbation for all tokens
        u = rand_gen_fn(output.shape).to(output.dtype)
        
        ### shared perturbation within a sequence
        # u = rand_gen_fn((output.size(0), output.size(-1))).to(output.dtype)
        # u = u.unsqueeze(1).expand(-1, output.size(1), -1)
        
        ### LRT
        # u = 
        
        ### power law scaling, sigma = C j^{-1/2}, u = sigma * e
        # scale = torch.arange(1, u.size(1)+1).pow(-0.1).to(output.dtype).to(output.device)
        # u *= scale.reshape(1, -1, 1)
        
        # torch.set_rng_state(state)
        
        if token_idx is not None: 
            u_mask = torch.zeros_like(u)
            u_mask[:, token_idx, :] = 1
            u = u * u_mask
        
        if mask is not None:
            u = u * mask
        
        # print(f'u tokenwise norm {u.norm(dim=(0,2))}')
            
        module.perturbation = u
        
        # output += sigma * perturbation
        return input + sigma * u
    return fwd_hook
      
class ZO_Estim_PZO(ZO_Estim_Base):
    """
    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
    """
    
    def __init__(
            self,
            **kwargs
        ):
        super().__init__(**kwargs)
            
    ##################
    ### residual implementation ###
    ##################
    
    def get_pseudo_actv_ZO_gradient(self):

        ### Generate random perturbation with the same shape as the parameter
        ### Add perturbation to the parameter
        ### Estimate gradient
        
        if self.sample_method == 'coord_basis':
            raise NotImplementedError
      
        actv_dim = self.ZO_dimension
        n_sample = self.n_sample
        
        # (bz, 1, 1, seq_len) -> (seq_len, bz, 1)
        # mask = self.obj_fn.get_mask()
        mask = None
        
        
        # =============================== Clean Forward ===============================
        if self.estimate_method == 'forward':            
            with torch.no_grad():
                # outputs, loss = self.obj_fn()
                hidden_states = self.obj_fn.get_hidden_states()
            
            hidden_states = hidden_states.detach()
            output = hidden_states
            output_grad = self.obj_fn.get_grad_hidden_states(hidden_states)
            
            self.model.zero_grad()
            self.forward_counter += 1
        else:
            pass
        
        with torch.no_grad():
            ### Perturbed forwards
            for i in range(n_sample):
                fwd_hook_handle_list = []
                ### Add perturbation to all activations
                for splited_layer in self.splited_layer_list:
                    ### USE WHEN perturb next layer mode
                    # if splited_layer.perturb_layer is not None:
                    
                    splited_layer.ZO_random_seed = torch.randint(0, 100000, (1,))
                    
                    ### add perturbation to the output
                    fwd_hook_add_perturbation = splited_layer.create_fwd_hook_add_perturbation(splited_layer.ZO_random_seed, self.sigma, self.rand_gen_fn, mask)
                    fwd_hook_handle = splited_layer.layer.register_forward_hook(fwd_hook_add_perturbation)
                    fwd_hook_handle_list.append(fwd_hook_handle)

                    ### add perturbation to the next layer input
                    # fwd_hook_add_perturbation = splited_layer.create_fwd_pre_hook_add_perturbation(splited_layer.ZO_random_seed, self.sigma, self.rand_gen_fn, mask)
                    # fwd_hook_handle = splited_layer.perturb_layer.register_forward_pre_hook(fwd_hook_add_perturbation)
                    # fwd_hook_handle_list.append(fwd_hook_handle)

                if self.estimate_method == 'one_point':
                    with torch.no_grad():
                        # outputs, loss = self.obj_fn()
                        hidden_states = self.obj_fn.get_hidden_states()
                    
                    hidden_states = hidden_states.detach()
                    pos_output = hidden_states
                    pos_output_grad = self.obj_fn.get_grad_hidden_states(pos_output)
                else:
                    # pos_output, _ = self.obj_fn(return_loss_reduction='pzo_nograd')
                    pos_output = self.obj_fn.get_hidden_states()
                
                self.model.zero_grad()
                
                for fwd_hook_handle in fwd_hook_handle_list:    
                    fwd_hook_handle.remove()
                    
                self.forward_counter += 1

                if self.estimate_method == 'antithetic':
                    fwd_hook_handle_list = []
                    ### Add perturbation to the parameter
                    for splited_layer in self.splited_layer_list:
                        ### USE WHEN perturb next layer mode
                        # if splited_layer.perturb_layer is not None:
                        ### add perturbation to the output
                        fwd_hook_add_perturbation = splited_layer.create_fwd_hook_add_perturbation(splited_layer.ZO_random_seed, -self.sigma, self.rand_gen_fn, mask)
                        fwd_hook_handle = splited_layer.layer.register_forward_hook(fwd_hook_add_perturbation)
                        fwd_hook_handle_list.append(fwd_hook_handle)
                        
                        ### add perturbation to the next layer input
                        # fwd_hook_add_perturbation = splited_layer.create_fwd_pre_hook_add_perturbation(splited_layer.ZO_random_seed, -self.sigma, self.rand_gen_fn, mask)
                        # fwd_hook_handle = splited_layer.perturb_layer.register_forward_pre_hook(fwd_hook_add_perturbation)
                        # fwd_hook_handle_list.append(fwd_hook_handle)

                    # neg_output, _ = self.obj_fn(return_loss_reduction='pzo_nograd')
                    neg_output = self.obj_fn.get_hidden_states()
                    self.model.zero_grad()

                    for fwd_hook_handle in fwd_hook_handle_list:    
                        fwd_hook_handle.remove()
                    
                    self.forward_counter += 1
                    
                if self.estimate_method == 'one_point':
                    tilde_o = pos_output
                    output_grad = pos_output_grad
                elif self.estimate_method == 'forward':
                    tilde_o = pos_output - output
                elif self.estimate_method == 'antithetic':
                    tilde_o = (pos_output - neg_output) / 2.0
                
                batch_sz = tilde_o.shape[1]
                seq_len = tilde_o.shape[0]
                
                ### scaling factor
                scaling_factor = (1 / n_sample / batch_sz)

                ### No scaling
                if self.scale is None:
                    pass
                elif self.scale == 'sqrt_dim':
                    scaling_factor *= math.sqrt(batch_sz*n_sample / (batch_sz*n_sample+actv_dim-1))
                elif self.scale == 'dim':
                    scaling_factor *= (batch_sz*n_sample / (batch_sz*n_sample+actv_dim-1))    
                        
                ### estimate gradient
                for splited_layer in self.splited_layer_list:
                    ### USE WHEN perturb next layer mode
                    # if splited_layer.perturb_layer is not None:
                    u = splited_layer.layer.perturbation
                    
                    ### merge seq_len and batch_sz  
                    feedback_matrix = torch.einsum('...d,...m->dm', (u, tilde_o))
                    
                    ### each token to [CLS] token
                    # feedback_matrix = torch.einsum('sbd,bm->sdm', (u, tilde_o[0]))
                    
                    ### only merge batch_sz
                    # u = u.permute(1, 0, 2).reshape(batch_sz, -1)
                    # tilde_o = tilde_o.permute(1, 0, 2).reshape(batch_sz, -1)
                    # feedback_matrix = torch.einsum('bd,bm->bdm', (u, tilde_o))
                    
                    if float(self.config.pzo_momentum) > 0: 
                        if hasattr(splited_layer.layer, 'feedback_matrix') is False:
                            # splited_layer.layer.feedback_matrix = feedback_matrix
                            splited_layer.layer.register_buffer('feedback_matrix', feedback_matrix)
                        else:
                            splited_layer.layer.feedback_matrix = self.config.pzo_momentum * splited_layer.layer.feedback_matrix + (1 - self.config.pzo_momentum) * feedback_matrix
                            feedback_matrix = splited_layer.layer.feedback_matrix

                    ### init
                    if i == 0:
                        splited_layer.layer.ZO_grad_output = torch.zeros_like(u)

                    ### merge seq_len and batch_sz  
                    splited_layer.layer.ZO_grad_output += scaling_factor * torch.einsum('dm,...m->...d', (feedback_matrix, output_grad))
                    
                    ### each token to [CLS] token
                    # splited_layer.layer.ZO_grad_output += scaling_factor * torch.einsum('sdm,bm->sbd', (feedback_matrix, output_grad[0]))
                    
                    ### only merge batch_sz
                    # output_grad = output_grad.permute(1, 0, 2).reshape(batch_sz, -1)
                    # splited_layer.layer.ZO_grad_output += scaling_factor * torch.einsum('bdm,bm->bd', (feedback_matrix, output_grad)).reshape(batch_sz, seq_len, -1).permute(1, 0, 2)
        
        ### resiudal connect
        # for splited_layer in self.splited_layer_list:
        #     if splited_layer.perturb_layer is not None:
        #         splited_layer.layer.ZO_grad_output += output_grad
        #     else:
        #         splited_layer.layer.ZO_grad_output = output_grad
        
        return None
      
    
    def get_DFA_gradient(self):

        ### Generate random perturbation with the same shape as the parameter
        ### Add perturbation to the parameter
        ### Estimate gradient
        
        if self.sample_method == 'coord_basis':
            raise NotImplementedError
      
        for splited_layer in self.splited_layer_list:
            if hasattr(splited_layer, 'feedback_matrix') is False:
                # print('init feedback matrix')
                OUT_DIM = splited_layer.layer.output_shape[-1]
                # The random feedback matrix is uniformly sampled between [-1, 1)
                splited_layer.register_buffer('feedback_matrix', (torch.randn((OUT_DIM, OUT_DIM), device=self.device, dtype=self.dtype)* 2 - 1) * math.sqrt(3/OUT_DIM))
        
        # =============================== Forward ===============================
        with torch.no_grad():
            # outputs, loss = self.obj_fn()
            hidden_states = self.obj_fn.get_hidden_states()
        
        hidden_states = hidden_states.detach()
        output_grad = self.obj_fn.get_grad_hidden_states(hidden_states)
        
        self.model.zero_grad()
        self.forward_counter += 1
        
        with torch.no_grad():
            for splited_layer in self.splited_layer_list:
                if splited_layer.perturb_layer is not None:
                    ### Vanilla DFA
                    # splited_layer.layer.ZO_grad_output = torch.einsum('...d,dd->...d', (output_grad, splited_layer.feedback_matrix))
                    ### resiudal connect
                    splited_layer.layer.ZO_grad_output = output_grad + torch.einsum('...d,dd->...d', (output_grad, splited_layer.feedback_matrix))
                    ### k=0
                    # splited_layer.layer.ZO_grad_output = output_grad
                else:
                    splited_layer.layer.ZO_grad_output = output_grad
        
        return None
    
    def estimate_grad(self):
        self.get_pseudo_actv_ZO_gradient()
        # self.get_DFA_gradient()
        
        fwd_pre_hook_list = []
        bwd_pre_hook_list = []
        
        for splited_layer in self.splited_layer_list:
            if splited_layer.mode == 'actv':
                # fwd_pre_hook_list.append(splited_layer.layer.register_forward_pre_hook(fwd_pre_hook_detach_input))

                create_bwd_pre_hook_ZO_grad = getattr(splited_layer.layer, 'create_bwd_pre_hook_ZO_grad', default_create_bwd_pre_hook_ZO_grad)
                bwd_pre_hook_list.append(splited_layer.layer.register_full_backward_pre_hook(create_bwd_pre_hook_ZO_grad(splited_layer.layer.ZO_grad_output, DEBUG)))
        outputs, loss = self.obj_fn()
        loss.backward()
        
        # skip embedding gradient
        embedding = self.model.model.embed_tokens
        for param in embedding.parameters():
            param.grad = torch.zeros_like(param)
        
        for fwd_pre_hook in fwd_pre_hook_list:
            fwd_pre_hook.remove()
            
        for bwd_pre_hook in bwd_pre_hook_list:
            bwd_pre_hook.remove()
        
        return outputs, loss