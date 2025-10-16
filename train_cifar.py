import os
import time
import json
from math import prod
import wandb
import torch
from torch.nn import CrossEntropyLoss
from model import set_seed
from nn.cola_nn import cola_parameterize, get_model_summary_and_flops
import nn
from tqdm import tqdm
from scaling_mlps.data_utils import data_stats
from scaling_mlps.data_utils.dataloader import get_loader
from scaling_mlps.utils.config import config_to_name
from scaling_mlps.utils.metrics import topk_acc, real_acc, AverageMeter
from scaling_mlps.utils.optimizer import get_scheduler
from scaling_mlps.utils.parsers import get_training_parser
from einops import rearrange

import yaml
from easydict import EasyDict
import torch.nn.functional as F
import sys

# ===== ZO Gradient Estimator (New Implementation) =====
# Using clean ZO_grad_estimator with backward-compatible names
# Old ZO_Estim still works, but new implementation is recommended
from ZO_grad_estimator import ZOEstimator, ZOConfig, build_objective_function
from ZO_grad_estimator.utils import create_bwd_pre_hook_replace_grad

# Backward compatibility aliases for minimal code changes
build_ZO_Estim = lambda config, model: ZOEstimator(config, model)
build_obj_fn = build_objective_function
default_create_bwd_pre_hook_ZO_grad = create_bwd_pre_hook_replace_grad

# DEBUG=False  # Set to True to compare ZO gradients with true BP gradients
DEBUG=True  # When enabled, prints cosine similarity between ZO and BP gradients for all parameters

OUT_GRAD_DEBUG=False  # Set to True to compare ZO_grad_output with true output gradients (node perturbation only)
# OUT_GRAD_DEBUG=True  # When enabled, prints cosine similarity between ZO and BP output gradients for each layer


class TeeLogger:
    """Capture stdout to both terminal and a file"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def train(model, opt, scheduler, loss_fn, epoch, train_loader, ZO_Estim, args):
    start = time.time()
    model.train()

    total_acc, total_top5 = AverageMeter(), AverageMeter()
    total_loss = AverageMeter()
    total_aux_loss = AverageMeter()

    for step, (ims, targs) in enumerate(train_loader):
        if ZO_Estim is not None:
            ### ZO grad estimation
            obj_fn = build_obj_fn(ZO_Estim.config.obj_fn_type, model=model, ims=ims, targs=targs, loss_fn=loss_fn, args=args)
            ZO_Estim.update_objective(obj_fn)

            ### Weight Perturbation (WP)
            if ZO_Estim.perturb_params:
                # Set model to eval mode (disable dropout for ZO)
                model.eval()
                # Estimate gradients (assigns to param.grad)
                outputs, loss = ZO_Estim.estimate_grad()

                # DEBUG: Compare ZO gradients with true BP gradients
                if DEBUG:
                    # Save ZO gradients
                    zo_grads = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            zo_grads[name] = param.grad.clone()

                    # Compute true gradients via backpropagation
                    model.train()
                    model.zero_grad()
                    outputs_bp, loss_bp = obj_fn()
                    loss_bp.backward()

                    # Compare ZO vs BP gradients
                    print('\n=== DEBUG: ZO vs BP Gradient Comparison (Weight Perturbation) ===')
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in zo_grads:
                            zo_grad = zo_grads[name]
                            bp_grad = param.grad
                            if bp_grad is not None:
                                cos_sim = F.cosine_similarity(
                                    zo_grad.reshape(-1),
                                    bp_grad.reshape(-1),
                                    dim=0
                                )
                                zo_norm = torch.linalg.norm(zo_grad.reshape(-1))
                                bp_norm = torch.linalg.norm(bp_grad.reshape(-1))
                                print(f'{name:60s} | cos_sim={cos_sim:.4f} | '
                                      f'ZO_norm={zo_norm:.4e} | BP_norm={bp_norm:.4e}')

                    # Restore ZO gradients
                    for name, param in model.named_parameters():
                        if name in zo_grads:
                            param.grad = zo_grads[name]

            ### Node Perturbation (NP)
            elif ZO_Estim.perturb_layers:
                # Set model to eval mode
                model.eval()
                # Estimate ZO_grad_output
                ZO_Estim.estimate_grad()

                # OUT_GRAD_DEBUG: Compare ZO_grad_output with true output gradients
                if OUT_GRAD_DEBUG:
                    # Save ZO_grad_output for each layer
                    zo_grad_outputs = {}
                    for perturb_layer in ZO_Estim.perturb_layers:
                        if perturb_layer.mode == 'actv' and hasattr(perturb_layer.layer, 'ZO_grad_output'):
                            zo_grad_outputs[perturb_layer.name] = perturb_layer.layer.ZO_grad_output.clone()

                    # Compute true output gradients via BP
                    # Register hooks to capture true output gradients
                    true_grad_outputs = {}

                    def make_grad_hook(layer_name):
                        def hook(module, grad_input, grad_output):
                            # grad_output[0] is the gradient w.r.t. the output
                            if grad_output[0] is not None:
                                true_grad_outputs[layer_name] = grad_output[0].detach().clone()
                        return hook

                    grad_hooks = []
                    for perturb_layer in ZO_Estim.perturb_layers:
                        if perturb_layer.mode == 'actv':
                            hook = perturb_layer.layer.register_full_backward_hook(
                                make_grad_hook(perturb_layer.name)
                            )
                            grad_hooks.append(hook)

                    # Run BP to get true output gradients
                    model.train()
                    model.zero_grad()
                    outputs_bp, loss_bp = obj_fn()
                    loss_bp.backward()

                    # Remove gradient capture hooks
                    for hook in grad_hooks:
                        hook.remove()

                    # Compare ZO_grad_output vs true output gradients
                    print('\n=== OUT_GRAD_DEBUG: ZO vs BP Output Gradient Comparison ===')
                    for layer_name in zo_grad_outputs.keys():
                        if layer_name in true_grad_outputs:
                            zo_grad = zo_grad_outputs[layer_name]
                            bp_grad = true_grad_outputs[layer_name]

                            cos_sim = F.cosine_similarity(
                                zo_grad.reshape(-1),
                                bp_grad.reshape(-1),
                                dim=0
                            )
                            zo_norm = torch.linalg.norm(zo_grad.reshape(-1))
                            bp_norm = torch.linalg.norm(bp_grad.reshape(-1))

                            print(f'{layer_name:60s} | cos_sim={cos_sim:.4f} | '
                                  f'ZO_norm={zo_norm:.4e} | BP_norm={bp_norm:.4e}')

                # Pseudo-NP: Use backward hooks to get param gradients
                model.train()
                bwd_pre_hook_list = []
                for perturb_layer in ZO_Estim.perturb_layers:
                    if perturb_layer.mode == 'actv':
                        create_bwd_pre_hook_ZO_grad = getattr(
                            perturb_layer.layer,
                            'create_bwd_pre_hook_ZO_grad',
                            default_create_bwd_pre_hook_ZO_grad
                        )
                        bwd_pre_hook_list.append(
                            perturb_layer.layer.register_full_backward_pre_hook(
                                create_bwd_pre_hook_ZO_grad(perturb_layer.layer.ZO_grad_output, DEBUG)
                            )
                        )

                # Forward and backward
                outputs, loss = obj_fn()
                loss.backward()

                # Remove hooks
                for bwd_pre_hook in bwd_pre_hook_list:
                    bwd_pre_hook.remove()

                # DEBUG: Compare ZO gradients with true BP gradients
                if DEBUG:
                    # Save ZO gradients
                    zo_grads = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            zo_grads[name] = param.grad.clone()

                    # Compute true gradients via backpropagation (without hooks)
                    model.zero_grad()
                    outputs_bp, loss_bp = obj_fn()
                    loss_bp.backward()

                    # Compare ZO vs BP gradients
                    print('\n=== DEBUG: ZO vs BP Gradient Comparison (Node Perturbation) ===')
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in zo_grads:
                            zo_grad = zo_grads[name]
                            bp_grad = param.grad
                            if bp_grad is not None:
                                cos_sim = F.cosine_similarity(
                                    zo_grad.reshape(-1),
                                    bp_grad.reshape(-1),
                                    dim=0
                                )
                                zo_norm = torch.linalg.norm(zo_grad.reshape(-1))
                                bp_norm = torch.linalg.norm(bp_grad.reshape(-1))
                                print(f'{name:60s} | cos_sim={cos_sim:.4f} | '
                                      f'ZO_norm={zo_norm:.4e} | BP_norm={bp_norm:.4e}')

                    # Restore ZO gradients
                    for name, param in model.named_parameters():
                        if name in zo_grads:
                            param.grad = zo_grads[name]

            ### For metric loading
            preds = outputs.logits
            if args.mixup > 0:
                targs_perm = targs[:, 1].long()
                weight = targs[0, 2].squeeze()
                targs = targs[:, 0].long()
                if weight != -1:
                    pass
                else:
                    targs_perm = None
            else:
                if args.ar_modeling:
                    targs = rearrange(ims, 'b c h w -> (b h w c)')
                    preds = preds[:, :-1].reshape(-1, preds.shape[-1])  # nothing to predict after the last pixel
                loss = loss_fn(preds, targs)
                targs_perm = None
                
            # load balancing loss
            aux_losses = []
            for name, module in model.named_modules():
                if 'moe_gate' in name:
                    aux_losses.append(module.load_balancing_loss)
            aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else 0

        ### Standard Backpropagation-based training
        else:
            preds = model(ims)
            if args.mixup > 0:
                targs_perm = targs[:, 1].long()
                weight = targs[0, 2].squeeze()
                targs = targs[:, 0].long()
                if weight != -1:
                    loss = loss_fn(preds, targs) * weight + loss_fn(preds, targs_perm) * (1 - weight)
                else:
                    loss = loss_fn(preds, targs)
                    targs_perm = None
            else:
                if args.ar_modeling:
                    targs = rearrange(ims, 'b c h w -> (b h w c)')
                    preds = preds[:, :-1].reshape(-1, preds.shape[-1])  # nothing to predict after the last pixel
                loss = loss_fn(preds, targs)
                targs_perm = None

            # load balancing loss
            aux_losses = []
            spec_penalties = []
            for name, module in model.named_modules():
                if 'moe_gate' in name:
                    aux_losses.append(module.load_balancing_loss)
                if hasattr(module, 'natural_norm'):
                    spec_penalties.append(module.natural_norm**2)
            aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else 0
            spec_penalty = sum(spec_penalties) / len(spec_penalties) if spec_penalties else 0
            loss += aux_loss * args.aux_loss_weight
            loss += spec_penalty * args.spec_penalty_weight

            loss = loss / args.accum_steps
            loss.backward()
        
        acc, top5 = topk_acc(preds, targs, targs_perm, k=5, avg=True)
        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])

        if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            opt.zero_grad()

        total_loss.update(loss.item() * args.accum_steps, ims.shape[0])
        total_aux_loss.update(aux_loss * args.accum_steps, ims.shape[0])

    end = time.time()

    scheduler.step()

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
        total_loss.get_avg(percentage=False),
        total_aux_loss.get_avg(percentage=False),
        end - start,
    )


@torch.no_grad()
def test(model, loader, loss_fn, args):
    start = time.time()
    model.eval()
    total_acc, total_top5, total_loss = AverageMeter(), AverageMeter(), AverageMeter()

    for ims, targs in loader:
        preds = model(ims)
        if args.ar_modeling:
            targs = rearrange(ims, 'b c h w -> (b h w c)')
            preds = preds[:, :-1].reshape(-1, preds.shape[-1])  # nothing to predict after the last pixel
        if args.dataset != 'imagenet_real':
            acc, top5 = topk_acc(preds, targs, k=5, avg=True)
            loss = loss_fn(preds, targs).item()
        else:
            acc = real_acc(preds, targs, k=5, avg=True)
            top5 = 0
            loss = 0

        total_acc.update(acc, ims.shape[0])
        total_top5.update(top5, ims.shape[0])
        total_loss.update(loss)

    end = time.time()

    return (
        total_acc.get_avg(percentage=True),
        total_top5.get_avg(percentage=True),
        total_loss.get_avg(percentage=False),
        end - start,
    )


def main(args):
    set_seed(args.seed)
    # Use mixed precision matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_shape = (1, 3, args.crop_resolution, args.crop_resolution)
    model_builder = getattr(nn, args.model)
    base_ffn_expansion = 4  # equivalent to specifying a constant LR multuplier in μP. 4 works well for ViT.
    base_config = dict(dim_in=prod(input_shape), dim_out=args.num_classes, depth=args.depth, width=args.width,
                       num_ffn_experts=args.num_ffn_experts, ffn_expansion=base_ffn_expansion, patch_size=args.patch_size,
                       in_channels=args.in_channels, shuffle_pixels=args.shuffle_pixels, heads=args.heads, dim_head=args.dim_head,
                       attn_mult=args.attn_mult, output_mult=args.output_mult, emb_mult=args.emb_mult, layer_norm=args.layer_norm,
                       mlp_activation=args.mlp_activation)
    # target config
    target_config = base_config.copy()
    args.width = int(args.width * args.scale_factor)  # we update args.width to have it logged in wandb
    target_config['width'] = args.width
    target_config['ffn_expansion'] = args.ffn_expansion

    # additional LR multipliers
    def extra_lr_mult_fn(param_name):
        if 'to_patch_embedding' in param_name or 'input_layer' in param_name:
            return args.input_lr_mult
        elif 'op_params.0' in param_name:
            print(f'scaling {param_name} LR by {args.lr_mult_1}')
            return args.lr_mult_1
        elif 'op_params.1' in param_name:
            print(f'scaling {param_name} LR by {args.lr_mult_2}')
            return args.lr_mult_2
        else:
            return 1

    def extra_init_mult_fn(param_name):
        if 'op_params.0' in param_name:
            print(f'scaling {param_name} std by {args.init_mult_1}')
            return args.init_mult_1
        elif 'op_params.1' in param_name:
            print(f'scaling {param_name} std by {args.init_mult_2}')
            return args.init_mult_2
        else:
            return 1

    def zero_init_fn(weight, name):
        return hasattr(weight, 'zero_init') and weight.zero_init

    # CoLA structure
    struct = args.struct
    fact_cls = None
    if struct.startswith("einsum"):
        fact_cls = select_factorizer(name=args.fact)
        fact_cls = fact_cls(cores_n=args.cores_n, int_pow=args.int_pow)
        fact_cls.sample(expr=args.expr)
        base_config["fact_cls"] = fact_cls
        target_config["fact_cls"] = fact_cls
    cola_kwargs = dict(tt_cores=args.tt_cores, tt_rank=args.tt_rank, num_blocks=args.num_blocks, rank_frac=args.rank_frac,
                       fact_cls=fact_cls, expr=args.expr, init_type=args.init_type, do_sgd_lr=args.optimizer == "sgd",
                       low_rank_activation=args.low_rank_activation, actv_between=args.actv_between, actv_output=args.actv_output)

    # Create unique identifier early to set up logging
    run_name = config_to_name(args)
    path = os.path.join(args.checkpoint_folder, run_name)
    if not os.path.exists(path):
        os.makedirs(path)

    # ========== Set up logging to capture initialization ==========
    # Redirect stdout to both terminal and log file (W&B will capture stdout automatically)
    log_filename = os.path.join(path, 'model_init_log.txt')
    tee_logger = TeeLogger(log_filename)
    old_stdout = sys.stdout
    sys.stdout = tee_logger

    print("="*80)
    print("MODEL INITIALIZATION LOG")
    print("="*80)
    print(f"Run name: {run_name}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Structure: {struct}")
    print(f"Base config: {base_config}")
    print(f"Target config: {target_config}")
    print(f"CoLA kwargs: {cola_kwargs}")
    print("="*80)

    # initialize scaled up model with some linear layers replaced by cola layers,
    # and create optimizer with appropriately scaled learning rates
    if args.use_wrong_mult:
        print("#### WARNING: using wrong mult ####")
    optim_kwargs = {"opt_name": args.optimizer}
    model, opt = cola_parameterize(model_builder, base_config, args.lr, target_config=target_config, struct=struct,
                                   layer_select_fn=args.layers, zero_init_fn=zero_init_fn, extra_lr_mult_fn=extra_lr_mult_fn,
                                   device=device, cola_kwargs=cola_kwargs, use_wrong_mult=args.use_wrong_mult, init_method=args.init_method,
                                   optim_kwargs=optim_kwargs)
    fake_input = torch.zeros(*input_shape).to('cuda')
    if args.ar_modeling:
        fake_input = fake_input.long()
    info = get_model_summary_and_flops(model, fake_input)
    # if struct == "einsum":
    #     info["cola_flops"] += fact_cls.flops
    #     print(fact_cls.layers)

    print("="*80)
    print("MODEL PARAMETERS")
    print("="*80)
    for name, param in model.named_parameters():
        print(f'{name:40s} | shape={str(param.shape):20s} | numel={param.numel():10d} | requires_grad={param.requires_grad}')
    print("="*80)

    # Stop capturing and close the log file
    sys.stdout = old_stdout
    tee_logger.close()
    print(f"Model initialization log saved to: {log_filename}")
    # ========== End of capture ==========

    scheduler = get_scheduler(opt, args.scheduler, **args.__dict__)
    loss_fn = CrossEntropyLoss(label_smoothing=args.smooth)
    
    # ================== ZO_Estim ======================
    ZO_Estim = None
    if args.ZO_config_path is not None:
        # Load config using new ZOConfig (backward compatible with EasyDict)
        ZO_config = ZOConfig.from_yaml(args.ZO_config_path)

        # Create ZO estimator (uses new ZO_grad_estimator)
        ZO_Estim = build_ZO_Estim(ZO_config, model=model)
    # ================== ZO_Estim ======================

    # Save config file
    with open(path + '/config.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Get the dataloaders
    local_batch_size = args.batch_size // args.accum_steps

    train_loader = get_loader(args.dataset, bs=local_batch_size, mode="train", augment=args.augment, dev=device,
                              num_samples=args.n_train, mixup=args.mixup, data_path=args.data_path,
                              data_resolution=args.resolution, crop_resolution=args.crop_resolution, ar_modeling=args.ar_modeling)

    test_loader = get_loader(args.dataset, bs=local_batch_size, mode="test", augment=False, dev=device, data_path=args.data_path,
                             data_resolution=args.resolution, crop_resolution=args.crop_resolution, ar_modeling=args.ar_modeling)

    if args.wandb:
        config = args.__dict__
        config.update(info)
        if struct.startswith("ein_expr"):
            config.update({"expr0": args.expr})
        if struct.startswith("einsum"):
            config.update(fact_cls.log_data())
            exprs = fact_cls.get_unique_ein_expr()
            formated_combined = {f"expr{idx}": f"{key}({val:d})" for idx, (key, val) in enumerate(exprs.items())}
            config.update(formated_combined)
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            tags=["pretrain", args.dataset],
        )

        # Upload model initialization log to wandb Files
        wandb.save(log_filename, base_path=args.checkpoint_folder)
        print(f"Model initialization log will be available in W&B Files tab")

    compute_per_epoch = info['cola_flops'] * len(train_loader) * args.batch_size

    prev_hs = None
    recent_train_accs = []
    recent_train_losses = []
    start_ep = 0
    for ep in (pb := tqdm(range(start_ep, args.epochs))):
        calc_stats = ep == start_ep or ep == args.epochs - 1 or (ep + 1) % args.calculate_stats == 0

        current_compute = compute_per_epoch * ep
        if ep == 0:  # skip first epoch
            train_acc, train_top5, train_loss, aux_loss, train_time = 0, 0, 0, 0, 0
        else:
            train_acc, train_top5, train_loss, aux_loss, train_time = train(model, opt, scheduler, loss_fn, ep, train_loader, ZO_Estim, args)
        if len(recent_train_accs) == 10:
            recent_train_accs.pop(0)
        recent_train_accs.append(train_acc)
        if len(recent_train_losses) == 10:
            recent_train_losses.pop(0)
        recent_train_losses.append(train_loss)

        if calc_stats:
            # model.hs = [[] for _ in range(len(model.hs))]  # clear the list
            model.clear_features()
            test_acc, test_top5, test_loss, test_time = test(model, test_loader, loss_fn, args)
            # get features on test set
            # hs = model.hs  # list of lists of tensors
            hs = model.get_features()
            hs = [torch.cat(h.buffer, dim=0) for h in hs]  # list of tensors
            if prev_hs is None:
                prev_hs = hs
            dhs = [hs[i] - prev_hs[i] for i in range(len(hs))]
            h_norm = [torch.norm(h, dim=1).mean() / h.shape[1]**0.5 for h in hs]  # should be O(1)
            dh_norm = [torch.norm(dh, dim=1).mean() / dh.shape[1]**0.5 for dh in dhs]  # should be O(1)
            prev_hs = hs

            if args.wandb:
                logs = {
                    "epoch": ep,
                    "train_acc": train_acc,
                    "train_acc_avg": sum(recent_train_accs) / len(recent_train_accs),
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "train_loss": train_loss,
                    "train_loss_avg": sum(recent_train_losses) / len(recent_train_losses),
                    "aux_loss": aux_loss,
                    "current_compute": current_compute,
                    "Inference time": test_time,
                }
                for i in range(len(h_norm)):
                    logs[f'h_{i}'] = h_norm[i].item()
                    logs[f'dh_{i}'] = dh_norm[i].item()
                # go through all params
                for name, p in model.named_parameters():
                    # if hasattr(p, 'rms'):
                    #     logs[f'rms/{name}'] = p.rms
                    if 'top_singular_vec' in name:
                        logs[f'v_rms/{name}'] = torch.sqrt((p**2).mean()).item()
                    if hasattr(p, 'scale'):
                        logs[f'scale/{name}'] = p.scale
                    if hasattr(p, 'x'):
                        logs[f'in/{name}'] = p.x
                    if hasattr(p, 'out'):
                        logs[f'out/{name}'] = p.out
                    if hasattr(p, 'ppl'):
                        logs[f'ppl/{name}'] = p.ppl
                    if hasattr(p, 'agg_ppl'):
                        logs[f'agg_ppl/{name}'] = p.agg_ppl
                for name, p in model.named_modules():
                    if hasattr(p, 'natural_norm'):
                        logs[f'natural_norm/{name}'] = p.natural_norm
                wandb.log(logs)
            pb.set_description(f"Epoch {ep}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")
        if torch.isnan(torch.tensor(train_loss)) and ep >= 5:
            break

    if args.save:
        torch.save(
            model.state_dict(),
            path + "/final_checkpoint.pt",
        )


if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()

    args.num_classes = data_stats.CLASS_DICT[args.dataset]

    if args.n_train is None:
        args.n_train = data_stats.SAMPLE_DICT[args.dataset]

    if args.crop_resolution is None:
        args.crop_resolution = args.resolution

    main(args)
