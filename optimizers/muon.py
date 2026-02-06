## Muon code from Moonlight
## https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
import torch
from functools import partial
import math
import warnings
from .polar_express import PolarExpress, FastApplyPolarExpress

@torch.compile
def jiacheng(G, steps):
    """
    Jiacheng optimized polynomials
    """
    assert len(G.shape) >= 2
    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    if steps > len(abc_list):
        steps = len(abc_list)
    for a, b, c in abc_list[:steps]:
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315) 
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X


@torch.compile
def svd_exact_polar(G, _, cutoff=None, reverse=False):
    """
    Exact polar factorization via SVD
    """
    assert len(G.shape) >= 2
    U, Sigma, Vh = torch.linalg.svd(G.to(torch.float32), full_matrices=False)
    if cutoff is None:
        return (U @ Vh).to(G.dtype)
    else:
        Sigma = ((Sigma / Sigma.max()) >= cutoff).to(G.dtype)  # zero out small singular values
        if reverse: Sigma = 2*Sigma - 1
        return (U @ torch.diag(Sigma) @ Vh).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
        adjust_lr_for_muon: Whether to adjust the learning rate for Muon parameters based on their shape. default, jordan, kimi
    """
    def __init__(self,
                 named_params,
                 lr=1e-3,
                 weight_decay=0.1,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 rms_scaling=True,
                 nuclear_scaling=False,
                 polar_method="polarexpress",
                 adamw_betas=(0.95, 0.95),
                 adamw_eps=1e-8,
                 split_heads=False,
                 split_qkv=False,
                 nheads=None,
                 adjust_lr_method="default",
                 structured_adjust_lr_method="default",
                 structured_ortho_method="default",
                 enable_mup_retraction=False,
                 polar_args={},
                 polar_params=None
                ):
        """
        Arguments:
            polar_method: The name of the polar factorization method to use (e.g., "NewtonSchultz", "Keller", "Pole") where PolE = PolarExpress
        """
        defaults = dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                rms_scaling=rms_scaling,
                nuclear_scaling=nuclear_scaling,
                adamw_betas=adamw_betas,
                adamw_eps=adamw_eps,
                adjust_lr_method=adjust_lr_method,
                structured_adjust_lr_method=structured_adjust_lr_method,
                structured_ortho_method=structured_ortho_method,
                enable_mup_retraction=enable_mup_retraction,
        )
        
        # print("EMBED TOKENS AND LM_HEAD ARE NOT HANDLED CORRECTLY FOR MUON, THEY SHOULD BE WITH ADAMW.")
        muon_params, muon_params_names = [], []
        adamw_params, adamw_params_names = [], []
        for name, p in named_params:
            is_excluded = any(excluded in name for excluded in ["embeddings", "embed_tokens", "wte", "lm_head", "wpe"])
            is_structured = hasattr(p, "d_in") and hasattr(p, "d_out")
            if not is_excluded and (p.ndim >= 2 or is_structured):
                muon_params.append(p)
                muon_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)
        params = list(muon_params)
        params.extend(adamw_params)
        self.split_heads = split_heads
        self.split_qkv = split_qkv
        if self.split_heads:
            assert nheads is not None, "nheads must be specified if split_heads is True"
            self.nheads = nheads
        super().__init__(params, defaults)
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
        # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
        self._special_structs = {"btt", "btt_actv", "tt"}
        for p, p_name in zip(muon_params, muon_params_names):
            self.state[p]["use_muon"] = True
            self.state[p]["lr_mult"] = 1.0
            cola_struct = getattr(p, "cola_struct", None)
            self.state[p]["cola_struct"] = cola_struct
            self.state[p]["is_special_struct"] = cola_struct in self._special_structs
            if p_name.endswith("attn.c_attn.weight"):
                self.state[p]["is_W_QKV"] = True
            elif p_name.endswith("attn.c_proj.weight"):
                self.state[p]["is_W_O"] = True

        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False
        
        # print muon configs
        print(f"[Muon] Initialized with the following settings: {defaults}")
        print("[Muon] Parameters optimized by Muon:")
        for name in muon_params_names:
            print(f"  {name}")
        print("[Muon] Parameters optimized by AdamW:")
        for name in adamw_params_names:
            print(f"  {name}")

        # Instantiate the polar factorization method
        self.polar_factorizer = self._initialize_polar_factorizer(polar_method, polar_args)

    def _initialize_polar_factorizer(self, polar_method, polar_args):
        """Initialize the polar factorization method based on the provided name and parameters."""
        if polar_method == "Keller":
            return zeropower_via_newtonschulz5  # Use the method directly
        elif polar_method == "Jiacheng":
            return jiacheng
        elif polar_method == "polarexpress":
            return PolarExpress 
        elif polar_method == "fast_polarexpress":
            return partial(FastApplyPolarExpress, restart_interval=3, shift_eps=1e-3)
        elif polar_method == "svd-exact":
            return partial(svd_exact_polar, cutoff=polar_args.get("svd_cutoff", None), reverse=polar_args.get("svd_reverse", False))
        else:
            raise ValueError(f"Unknown polar method: {polar_method}")

    def _estimate_top_singular_value(self, p, state, n_iters=5, eps=1e-6):
        mat = p.detach()
        if mat.ndim != 2:
            return None
        mat_f = mat.float()
        v = state.get("mup_retraction_v", None)
        if v is None or v.shape[0] != mat_f.shape[1] or v.device != mat_f.device:
            v = torch.randn(mat_f.shape[1], device=mat_f.device, dtype=mat_f.dtype)
        v = v / (v.norm() + eps)
        n_iters = max(1, n_iters)
        for _ in range(n_iters):
            u = mat_f @ v
            u_norm = u.norm() + eps
            u = u / u_norm
            v = mat_f.mT @ u
            v_norm = v.norm() + eps
            v = v / v_norm
        state["mup_retraction_v"] = v.detach()
        sigma = u @ (mat_f @ v)
        return sigma.abs()

    def _structured_orthogonalize(self, p, g, group):
        """Placeholder for structured-layer orthogonalization strategy."""
        cola_struct = self.state[p].get("cola_struct", None)
        cola_role = getattr(p, "cola_role", None)
        shape_info = getattr(p, "cola_shape_info", None)
        def batched_polar(x):
            try:
                return torch.vmap(lambda m: self.polar_factorizer(m, group["ns_steps"]))(x)
            except Exception:
                out = torch.empty_like(x)
                for i in range(x.shape[0]):
                    out[i] = self.polar_factorizer(x[i], group["ns_steps"])
                return out
        if cola_struct == "tt" and cola_role == "tt_core" and shape_info is not None:
            structured_ortho_method=group.get("structured_ortho_method", "corrected")
            n = shape_info["n"]
            m = shape_info["m"]
            rank_prev = shape_info["rank_prev"]
            rank_next = shape_info["rank_next"]
            if structured_ortho_method == "corrected":
                # g shape: (rank_prev, n, m, rank_next) -> (rank_prev*rank_next, n, m)
                g_view = g.permute(0, 3, 1, 2).reshape(rank_prev * rank_next, n, m)
                u_view = batched_polar(g_view)
                return u_view.reshape(rank_prev, rank_next, n, m).permute(0, 2, 3, 1)
            elif structured_ortho_method == "muP":
                # g shape: (rank_prev, n, m, rank_next) -> (rank_prev*n, rank_next*m)
                u_view = self.polar_factorizer(g.reshape(rank_prev*n, m*rank_next), group["ns_steps"])
                return u_view.reshape(rank_prev, n, m, rank_next)

        if cola_struct in {"btt", "btt_actv"} and cola_role in {"btt_r", "btt_l"} and shape_info is not None:
            structured_ortho_method=group.get("structured_ortho_method", "default")
            if cola_role == "btt_r":
                m = shape_info["m"]
                n = shape_info["n"]
                rank = shape_info["rank"]
                b = shape_info["b"]
                
                if structured_ortho_method == "wrong":
                    ### wrong
                    # g shape: (b, m, rank * n) -> (b, m, rank, n) -> (b*rank, m, n)
                    g_view = g.view(b, m, rank, n).permute(0, 2, 1, 3).reshape(b * rank, m, n)
                    u_view = batched_polar(g_view)
                    return u_view.reshape(b, rank, m, n).permute(0, 2, 1, 3).reshape(b, m, rank * n)
                elif structured_ortho_method == "default":
                    ### corrected
                    # g shape: (b, m, rank * n) -> (b, m, rank, n) -> (b*n, rank, m)
                    g_view = g.view(b, m, rank, n).permute(0, 3, 2, 1).reshape(b * n, rank, m)
                    u_view = batched_polar(g_view)
                    return u_view.reshape(b, n, rank, m).permute(0, 3, 2, 1).reshape(b, m, rank * n)
                elif structured_ortho_method == "muP":
                    ### muP
                    # g shape: (b, m, rank * n) -> (b, din, dout)
                    return batched_polar(g)

            else:  # btt_l
                a = shape_info["a"]
                rank = shape_info["rank"]
                b = shape_info["b"]
                n = shape_info["n"]
                
                if structured_ortho_method == "wrong":
                    ### wrong
                    # g shape: (a, rank * b, n) -> (a, rank, b, n) -> (a*rank, b, n)
                    g_view = g.view(a, rank, b, n).reshape(a * rank, b, n)
                    u_view = batched_polar(g_view)
                    return u_view.reshape(a, rank, b, n).reshape(a, rank * b, n)
                elif structured_ortho_method == "default":
                    ### corrected
                    # g shape: (a, rank * b, n) -> (a, rank, b, n) -> (a* b, n, rank)
                    g_view = g.view(a, rank, b, n).permute(0,2,3,1).reshape(a * b, n, rank)
                    u_view = batched_polar(g_view)
                    return u_view.reshape(a, b, n, rank).permute(0, 3, 1, 2).reshape(a, rank * b, n)
                elif structured_ortho_method == "muP":
                    ### muP
                    # g shape: (a, rank * b, n) -> (b, din, dout)
                    return batched_polar(g)
        # TODO: implement per-structure orthogonalization for low_rank/low_rank_actv.
        return self.polar_factorizer(g, group["ns_steps"])

    def _structured_lr_scaling(self, p, adjusted_lr, adjust_lr_method="default"):
        """Placeholder for structured-layer LR scaling strategy."""
        cola_struct = self.state[p].get("cola_struct", None)
        cola_role = getattr(p, "cola_role", None)
        shape_info = getattr(p, "cola_shape_info", None)
        if cola_struct == "tt" and cola_role == "tt_core" and shape_info is not None:
            n = shape_info["n"]
            m = shape_info["m"]
            tt_rank = shape_info["tt_rank"]
            rank_prev = shape_info["rank_prev"]
            rank_next = shape_info["rank_next"]
            if tt_rank <= 0:
                raise ValueError(f"Invalid tt_rank: {tt_rank}")
            if adjust_lr_method == "default":
                scale = (1.0 / rank_prev) * math.sqrt(m / n)
            elif adjust_lr_method == "muP":
                scale = math.sqrt((rank_next * m) / (rank_prev * n))
            return adjusted_lr * scale
        if cola_struct in {"btt", "btt_actv"} and cola_role in {"btt_r", "btt_l"} and shape_info is not None:
            if cola_role == "btt_r":
                rank = shape_info["rank"]
                b = shape_info["b"]
                n = shape_info["n"]
                m = shape_info["m"]

                if adjust_lr_method == "default":
                    ### wrong
                    # scale = math.sqrt(rank / b)

                    ### corrected
                    scale = math.sqrt(rank / m)
                elif adjust_lr_method == "by_sqrt_b":
                    scale = math.sqrt(rank / m / b)
                elif adjust_lr_method == "by_b":
                    scale = math.sqrt(rank / m) / b
                elif adjust_lr_method == "mup_btt":
                    scale = math.sqrt(rank / m / n)
                elif adjust_lr_method == "mup_btt_new":
                    scale = math.sqrt(rank / m)
                elif adjust_lr_method == "muP":
                    scale = math.sqrt(rank * n / m)
                elif adjust_lr_method == "kimi":
                    scale = 0.2 *math.sqrt(max(rank, b))
                else:
                    raise ValueError(f"Unknown adjust_lr_method: {adjust_lr_method}")
                
                return adjusted_lr * scale
            else: # btt_l
                a = shape_info["a"]
                rank = shape_info["rank"]
                n = shape_info["n"]
                b = shape_info["b"]
                
                if adjust_lr_method == "default":
                    ### wrong
                    # scale = math.sqrt(a / rank)
                    
                    ### corrected
                    scale = math.sqrt(n / rank)
                elif adjust_lr_method == "by_sqrt_b":
                    scale = math.sqrt(n / rank / b)
                elif adjust_lr_method == "by_b":
                    scale = math.sqrt(n / rank) / b
                elif adjust_lr_method == "mup_btt":
                    scale = math.sqrt(n / rank / a)
                elif adjust_lr_method == "mup_btt_new":
                    scale = math.sqrt(n / rank) / a
                elif adjust_lr_method == "muP":
                    scale = math.sqrt(n / (rank * b))
                elif adjust_lr_method == "kimi":
                    scale = 0.2 *math.sqrt(max(a, rank))
                else:
                    raise ValueError(f"Unknown adjust_lr_method: {adjust_lr_method}")
                return adjusted_lr * scale
            
        # TODO: implement per-structure LR scaling for low_rank/low_rank_actv.
        return adjusted_lr

    def adjust_lr_for_muon(self, lr, rms_scaling, nuclear_scaling, param_shape, grad, grad_sign, fan_in=None, fan_out=None, adjust_lr_method="default"):
        scale = 1.0
        if rms_scaling:
            if fan_in is None or fan_out is None:
                fan_out, fan_in = param_shape[:2]
            if adjust_lr_method == "default":
                scale *= math.sqrt(fan_out / fan_in)
            elif adjust_lr_method == "jordan":
                scale *= math.sqrt(max(1, fan_out / fan_in))
            elif adjust_lr_method == "kimi":
                scale *= 0.2 * math.sqrt(max(fan_out, fan_in))
            else:
                raise ValueError(f"Unknown adjust_lr_method: {adjust_lr_method}")
        if nuclear_scaling:
            scale *= torch.trace(grad.T @ grad_sign)
        return lr * scale

    def step(self, closure=None):
        """Perform a single optimization step.
            Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
"""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                        
        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                g = p.grad
                if g is None:
                    continue
                reshape_back = False
                use_special = self.state[p].get("is_special_struct", False)
                if (g.ndim > 2) and not (self.split_heads) and not use_special:
                    old_shape = g.shape
                    g = g.view(g.size(0), -1)
                    reshape_back = True

                assert g is not None
                
                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                if self.split_heads and self.state[p].get("is_W_QKV", False):
                    # For W_QKV, we split the gradients into 3 heads and process them separately
                    # print("before", g.shape, self.nheads)
                    old_shape = g.shape
                    g = g.reshape(3 * self.nheads, g.shape[0] // (3 * self.nheads), g.shape[1])
                    # print("after", g.shape)
                elif self.split_heads and self.state[p].get("is_W_O", False) and self.split_heads:
                    # print("before", g.shape, self.nheads)
                    old_shape = g.shape
                    g = g.reshape(g.shape[0], self.nheads, g.shape[1] // self.nheads).transpose(0, 1)
                    # print("after", g.shape)
                    # For W_O, we split the gradients into 3 heads and process them separately

                # Use the selected polar factorization method
                if self.split_qkv and self.state[p].get("is_W_QKV", False) and not self.split_heads:
                    if g.shape[0] % 3 != 0:
                        raise ValueError(f"Expected fused QKV shape with dim0 divisible by 3, got {g.shape}")
                    qkv_dim = g.shape[0] // 3
                    g_q, g_k, g_v = g[:qkv_dim], g[qkv_dim:2 * qkv_dim], g[2 * qkv_dim:]
                    u_q = self._structured_orthogonalize(p, g_q, group) if use_special else self.polar_factorizer(g_q, group["ns_steps"])
                    u_k = self._structured_orthogonalize(p, g_k, group) if use_special else self.polar_factorizer(g_k, group["ns_steps"])
                    u_v = self._structured_orthogonalize(p, g_v, group) if use_special else self.polar_factorizer(g_v, group["ns_steps"])
                    u = torch.cat([u_q, u_k, u_v], dim=0)
                    
                    # scale update
                    fan_in = getattr(p, "d_in", None)
                    fan_out = getattr(p, "d_out", None)
                    adjusted_lr = self.adjust_lr_for_muon(
                        lr,
                        group["rms_scaling"],
                        group["nuclear_scaling"],
                        g_q.shape,
                        g.bfloat16(),  # convert to float16 to be compatible with u
                        u,
                        fan_in=fan_in,
                        fan_out=fan_out,
                        adjust_lr_method=group.get("adjust_lr_method", "default")
                    )
                else:
                    u = self._structured_orthogonalize(p, g, group) if use_special else self.polar_factorizer(g, group["ns_steps"])
                    
                    if self.split_heads and self.state[p].get("is_W_QKV", False):
                        g = g.reshape(old_shape)
                        u = u.reshape(old_shape)
                    elif self.split_heads and self.state[p].get("is_W_O", False):
                        g = g.transpose(0, 1).reshape(old_shape)
                        u = u.transpose(0, 1).reshape(old_shape)
                    
                    # scale update
                    fan_in = getattr(p, "d_in", None)
                    fan_out = getattr(p, "d_out", None)
                    adjusted_lr = self.adjust_lr_for_muon(
                        lr,
                        group["rms_scaling"],
                        group["nuclear_scaling"],
                        p.shape,
                        g.bfloat16(),  # convert to float16 to be compatible with u
                        u,
                        fan_in=fan_in,
                        fan_out=fan_out,
                        adjust_lr_method=group.get("adjust_lr_method", "default")
                    )

                # apply weight decay
                if use_special:
                    adjusted_lr = self._structured_lr_scaling(
                        p,
                        adjusted_lr,
                        adjust_lr_method=group.get(
                            "structured_adjust_lr_method",
                            group.get("adjust_lr_method", "default"),
                        ),
                    )
                if reshape_back:
                    u = u.view(old_shape)
                p.data.mul_(1 - adjusted_lr * weight_decay)
                
                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

                if group.get("enable_mup_retraction", False) and (p.ndim == 2) and not use_special:
                    s1 = self._estimate_top_singular_value(p, state)
                    if s1 is not None and torch.isfinite(s1) and s1 > 0:
                        p.data.mul_(1.0 / s1)
                
            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)
                    
        return loss
