import os
import sys
import time
import gc
from typing import Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from gptq.gptq import *
from utils.model_utils import *
from gptq.quant import *
from evaluater import ppl_eval
from utils.mixed_precision import (
    apply_two_path_quantization,
    apply_non_svd_fp16_cast,
    apply_non_svd_int8_quantization,
    apply_embed_tokens_fp16,
    apply_embed_tokens_int8_quantization,
    build_pair_whiten_inv,
    calibrate_component_sigma,
    clear_mixed_precision_runtime_cache,
    collect_kfac_stats_block_b,
    collect_kfac_stats_diagonal,
    collect_kfac_stats_sigma_full,
    compute_component_importance_block_b,
    compute_component_importance,
    compute_sigma_fisher_full,
    discover_low_rank_pairs,
    solve_budgeted_topk,
    solve_budgeted_topk_quadratic,
    solve_budgeted_multilevel_quadratic,
    set_mixed_precision_runtime_cache_policy,
    strip_original_low_rank_weights,
    MP_STATE_DROP,
    MP_STATE_LOW,
    MP_STATE_HIGH,
)

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)


def _base_weight_bits_from_dtype(dtype_name: str) -> float:
    if dtype_name == "fp32":
        return 32.0
    if dtype_name in {"fp16", "bf16"}:
        return 16.0
    return 16.0


def _infer_llama_family_from_paths(model_path: str, tokenizer_path: str) -> str:
    text = f"{model_path or ''} {tokenizer_path or ''}".lower()
    if "llama-2" in text or "llama_2" in text:
        return "llama2"
    if "llama-3" in text or "llama_3" in text or "llama3" in text:
        return "llama3"
    if "llama" in text:
        return "llama"
    return "unknown"


def _resolve_llama_family(model_path: str, tokenizer_path: str, family_arg: str) -> str:
    if family_arg and family_arg != "auto":
        return family_arg
    return _infer_llama_family_from_paths(model_path, tokenizer_path)


def _pack_int4_signed_rows(q: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Pack int8 tensor in [-8, 7] into uint8 (2 values per byte) along dim=1.
    Returns packed tensor and original column count.
    """
    if q.ndim != 2:
        raise ValueError(f"Expected 2D tensor for int4 packing, got shape={tuple(q.shape)}")
    q_i16 = q.to(dtype=torch.int16)
    q_u4 = (q_i16 + 8).clamp(0, 15).to(dtype=torch.uint8)
    rows, cols = q_u4.shape
    if cols % 2 == 1:
        pad = torch.full((rows, 1), 8, dtype=torch.uint8, device=q_u4.device)
        q_u4 = torch.cat([q_u4, pad], dim=1)
    lo = q_u4[:, 0::2]
    hi = q_u4[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()
    return packed, cols


def pack_lowbit_q_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Pack low-bit q buffers (ul_q/vl_q) from int8 to packed int4 representation.
    """
    out: Dict[str, torch.Tensor] = {}
    packed_tensors = 0
    before_bytes = 0
    after_bytes = 0
    for key, tensor in state_dict.items():
        if (
            key.endswith(".ul_q") or key.endswith(".vl_q")
        ) and isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int8 and tensor.ndim == 2 and tensor.numel() > 0:
            t_cpu = tensor.detach().to(device="cpu")
            t_min = int(t_cpu.min().item())
            t_max = int(t_cpu.max().item())
            if t_min >= -8 and t_max <= 7:
                packed, orig_cols = _pack_int4_signed_rows(t_cpu)
                out[f"{key}_packed4"] = packed
                out[f"{key}_orig_cols"] = torch.tensor([orig_cols], dtype=torch.int32)
                packed_tensors += 1
                before_bytes += int(t_cpu.numel())
                after_bytes += int(packed.numel()) + 4
                continue
        out[key] = tensor

    stats = {
        "packed_tensors": float(packed_tensors),
        "before_bytes": float(before_bytes),
        "after_bytes": float(after_bytes),
    }
    return out, stats

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_u_proj','self_attn.k_v_proj', 'self_attn.v_u_proj', 'self_attn.v_v_proj', 'self_attn.q_u_proj', 'self_attn.q_v_proj'],
                ['self_attn.o_u_proj', 'self_attn.o_v_proj'],
                ['mlp.up_u_proj', 'mlp.up_v_proj', 'mlp.gate_u_proj', 'mlp.gate_v_proj'],
                ['mlp.down_u_proj', 'mlp.down_v_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


@torch.no_grad()
def report_mixed_precision_allocation(
    pairs,
    alloc,
    high_bit: float,
    low_bit: float,
    drop_bit: float = 0.0,
    base_bits: float = None,
    sigma_eps: float = 1e-12,
):
    total = 0
    high = 0
    low = 0
    drop = 0
    active_total = 0
    active_high = 0
    active_low = 0
    active_drop = 0
    for p in pairs:
        if p.key not in alloc:
            continue
        mask = alloc[p.key]
        active = None
        try:
            u_norm = torch.linalg.vector_norm(p.u_module.weight.data.float(), dim=0)
            v_norm = torch.linalg.vector_norm(p.v_module.weight.data.float(), dim=1)
            active = (u_norm * v_norm) > float(sigma_eps)
        except Exception:
            active = None
        if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
            if active is not None:
                active = active.to(device=mask.device)
            total += int(mask.numel())
            high += int(mask.sum().item())
            low += int(mask.numel() - mask.sum().item())
            if active is not None and active.numel() == mask.numel():
                active_total += int(active.sum().item())
                active_high += int((mask & active).sum().item())
                active_low += int(((~mask) & active).sum().item())
        else:
            state = mask.to(dtype=torch.uint8)
            if active is not None:
                active = active.to(device=state.device)
            total += int(state.numel())
            high += int((state == MP_STATE_HIGH).sum().item())
            low += int((state == MP_STATE_LOW).sum().item())
            drop += int((state == MP_STATE_DROP).sum().item())
            if active is not None and active.numel() == state.numel():
                active_total += int(active.sum().item())
                active_high += int(((state == MP_STATE_HIGH) & active).sum().item())
                active_low += int(((state == MP_STATE_LOW) & active).sum().item())
                active_drop += int(((state == MP_STATE_DROP) & active).sum().item())
    avg_bits = 0.0
    if total > 0:
        high_bits = float(high_bit)
        low_bits = float(low_bit)
        drop_bits = float(drop_bit) if drop > 0 else 0.0
        avg_bits = (high * high_bits + low * low_bits + drop * drop_bits) / float(total)
    msg = f"Mixed-precision allocation: high={high}, low={low}, drop={drop}, total={total}, avg_bits={avg_bits:.4f}"
    if base_bits is not None and avg_bits > 0:
        est_ratio = float(base_bits) / avg_bits
        msg += f", est_lowrank_compression_ratio={est_ratio:.4f}x"
    print(msg)
    active_avg_bits = 0.0
    if active_total > 0:
        active_avg_bits = (
            active_high * float(high_bit) + active_low * float(low_bit) + active_drop * float(drop_bit)
        ) / float(active_total)
        active_drop_ratio = float(active_drop) / float(active_total)
        print(
            "Mixed-precision active allocation: "
            f"high={active_high}, low={active_low}, drop={active_drop}, total={active_total}, "
            f"drop_ratio={active_drop_ratio:.4f}, avg_bits={active_avg_bits:.4f}"
        )
    return {
        "high": high,
        "low": low,
        "drop": drop,
        "total": total,
        "avg_bits": avg_bits,
        "active_high": active_high,
        "active_low": active_low,
        "active_drop": active_drop,
        "active_total": active_total,
        "active_avg_bits": active_avg_bits,
    }


if __name__ == '__main__':
    import argparse
    from utils.data_utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path', type=str,
        help='path of the compressed model.'
    )
    parser.add_argument(
        '--model-dtype', type=str, default='fp16', choices=['fp32', 'fp16', 'bf16'],
        help='Model dtype cast before moving to device. fp16 is recommended for OOM mitigation.'
    )
    parser.add_argument(
        '--tokenizer_path', type=str, default=None,
        help='Optional tokenizer/model path or repo id when checkpoint does not include tokenizer.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--model_seq_len', type=int, default=2048,
        help='Sequence length used for calibration/evaluation loaders.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--mp-enable', action='store_true',
        help='Enable KFAC-weighted budgeted 8/4-bit two-path quantization on SVD low-rank pairs.'
    )
    parser.add_argument(
        '--mp-kfac-nsamples', type=int, default=8,
        help='Number of calibration mini-batches used to estimate diagonal KFAC factors.'
    )
    parser.add_argument(
        '--mp-kfac-grad-checkpointing', action='store_true',
        help='Enable gradient checkpointing during K-FAC stats collection to reduce activation memory.'
    )
    parser.add_argument(
        '--mp-kfac-layerwise', action='store_true',
        help='Collect K-FAC stats layer-by-layer (register hooks per layer group) to reduce peak memory.'
    )
    parser.add_argument(
        '--mp-kfac-mode', type=str, default='block_b', choices=['block_b', 'diag', 'sigma_full'],
        help='KFAC approximation used for component importance.'
    )
    parser.add_argument(
        '--mp-kfac-accum-device', type=str, default='cuda',
        help='Device used to accumulate K-FAC statistics matrices. Examples: cuda, cuda:1, cpu, auto. '
             'Use cuda:1 with --mp-kfac-proj-device cuda:1 to offload K-FAC statistics to a second GPU.'
    )
    parser.add_argument(
        '--mp-kfac-proj-device', type=str, default=None,
        help='Optional device for sigma_full projection buffers (U/V/W_inv) and hook-side projection matmuls, e.g., cuda:1.'
    )
    parser.add_argument(
        '--mp-explicit-sigma', action='store_true',
        help='Use explicit W=U*diag(sigma)*V^T parameterization for sigma-space Fisher and MP quantization.'
    )
    parser.add_argument(
        '--mp-sigma-eps', type=float, default=1e-12,
        help='Numerical epsilon used for explicit sigma decomposition and active component masking.'
    )
    parser.add_argument(
        '--mp-use-whiten', action='store_true',
        help='Use whitening matrix R^{-1} to whiten activations before sigma-space KFAC projection.'
    )
    parser.add_argument(
        '--mp-whiten-mat-path', type=str, default=None,
        help='Path to saved profiling matrix (*.pt) generated by SVDLLM step 1.'
    )
    parser.add_argument(
        '--mp-whiten-eps', type=float, default=1e-6,
        help='Stability epsilon added to whitening Cholesky factors before inversion.'
    )
    parser.add_argument(
        '--mp-kfac-block-size', type=int, default=128,
        help='Block size for Block-KFAC(B) covariance.'
    )
    parser.add_argument(
        '--mp-b-shrinkage', type=float, default=0.1,
        help='Shrinkage strength for Block-KFAC(B): B=(1-lambda)B+lambda*diag(B).'
    )
    parser.add_argument(
        '--mp-b-damp', type=float, default=1e-6,
        help='Diagonal damping added to each B block.'
    )
    parser.add_argument(
        '--mp-a-mode', type=str, default='adaptive', choices=['identity', 'diag', 'adaptive'],
        help='A-side approximation mode.'
    )
    parser.add_argument(
        '--mp-a-alpha', type=float, default=-1.0,
        help='Fixed alpha for adaptive A weighting. Negative means auto-estimate.'
    )
    parser.add_argument(
        '--mp-a-adaptive-tau', type=float, default=0.5,
        help='Tau used by auto alpha estimator in adaptive A mode.'
    )
    parser.add_argument(
        '--mp-sigma-mode', type=str, default='calibrated', choices=['proxy', 'calibrated'],
        help='Noise model used in budgeted allocation.'
    )
    parser.add_argument(
        '--mp-low-bit', type=int, default=4, choices=[2, 3, 4, 8],
        help='Low precision bit-width for residual components.'
    )
    parser.add_argument(
        '--mp-high-bit', type=int, default=8, choices=[4, 8, 16],
        help='High precision bit-width for dominant components.'
    )
    parser.add_argument(
        '--mp-avg-bit', type=float, default=4.5,
        help='Target average bit-width under budgeted top-k selection.'
    )
    parser.add_argument(
        '--mp-enable-drop', action='store_true',
        help='Enable joint truncation+quantization allocation with drop/low/high component states.'
    )
    parser.add_argument(
        '--mp-drop-bit', type=float, default=0.0,
        help='Bit-cost assigned to dropped components in budget accounting (typically 0).'
    )
    parser.add_argument(
        '--mp-min-keep-ratio', type=float, default=0.02,
        help='Minimum per-pair keep ratio (drop->low bootstrap) when joint drop allocation is enabled.'
    )
    parser.add_argument(
        '--mp-max-drop-ratio', type=float, default=-1.0,
        help='Maximum allowed active drop ratio in three-state allocation. '
             'Set <0 to auto-select by model family.'
    )
    parser.add_argument(
        '--mp-allow-drop-to-high', action='store_true',
        help='Allow direct drop->high upgrades in three-state allocation. '
             'By default, dropped components are restored through low/int4 first.'
    )
    parser.add_argument(
        '--mp-model-family', type=str, default='auto', choices=['auto', 'llama', 'llama2', 'llama3'],
        help='Model family hint for MP allocation safeguards. '
             'auto infers from model/tokenizer path; manually set when paths are ambiguous.'
    )
    parser.add_argument(
        '--mp-target-compression-ratio', type=float, default=0.0,
        help='If >0, auto-set mp_avg_bit from target low-rank compression ratio. '
             'Computed as mp_avg_bit = base_weight_bits / ratio, then clamped to [mp-low-bit, mp-high-bit].'
    )
    parser.add_argument(
        '--mp-disable-int8-kernel', action='store_true',
        help='Disable int8 GEMM kernel path in two-path quantized layers and force float fallback.'
    )
    parser.add_argument(
        '--mp-enable-int4-kernel', action='store_true',
        help='Enable native int4 kernel path (bitsandbytes Linear4bit) for 4-bit paths when available.'
    )
    parser.add_argument(
        '--mp-int4-quant-type', type=str, default='nf4', choices=['nf4', 'fp4'],
        help='Quant type for bitsandbytes 4-bit kernels.'
    )
    parser.add_argument(
        '--mp-persistent-runtime-cache', action='store_true',
        help='Keep MP runtime caches across calls for speed (uses much more GPU memory).'
    )
    parser.add_argument(
        '--mp-quantize-nonsvd-int8', action='store_true',
        help='[Deprecated] Quantize non-SVD nn.Linear modules to int8. Prefer --mp-nonsvd-precision int8.'
    )
    parser.add_argument(
        '--mp-nonsvd-precision', type=str, default='fp16', choices=['fp16', 'int8'],
        help='Non-SVD nn.Linear precision mode: fp16 or int8.'
    )
    parser.add_argument(
        '--mp-nonsvd-int8-exclude-lm-head', action='store_true',
        help='Exclude lm_head from non-SVD int8 quantization.'
    )
    parser.add_argument(
        '--mp-quantize-embed-int8', action='store_true',
        help='[Deprecated] Quantize embed_tokens to int8. Prefer --mp-embed-precision int8.'
    )
    parser.add_argument(
        '--mp-embed-precision', type=str, default='fp16', choices=['fp16', 'int8'],
        help='Embedding precision mode for embed_tokens: fp16 or int8.'
    )
    parser.add_argument(
        '--save-format', type=str, default='auto', choices=['auto', 'full', 'state_dict'],
        help='Checkpoint save format. auto -> state_dict when --mp-enable else full model object.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--eval-batch-size', type=int, default=4,
        help='Batch size used by ppl_eval.'
    )
    parser.add_argument('--DEV', type=str, default="cuda", help='device')

    args = parser.parse_args()
    base_weight_bits = _base_weight_bits_from_dtype(args.model_dtype)
    budget_min_bit = float(args.mp_drop_bit) if args.mp_enable_drop else float(args.mp_low_bit)
    if args.mp_target_compression_ratio is not None and args.mp_target_compression_ratio > 0:
        target_avg = base_weight_bits / float(args.mp_target_compression_ratio)
        clamped_avg = min(float(args.mp_high_bit), max(budget_min_bit, target_avg))
        if abs(clamped_avg - target_avg) > 1e-8:
            print(
                "Warning: target compression ratio implies avg bit outside [min, high]. "
                f"target_avg_bit={target_avg:.4f}, clamped={clamped_avg:.4f}"
            )
        args.mp_avg_bit = clamped_avg
        print(
            f"Auto-set mp_avg_bit={args.mp_avg_bit:.4f} from "
            f"mp_target_compression_ratio={args.mp_target_compression_ratio:.4f} "
            f"(base_weight_bits={base_weight_bits:.1f})"
        )
    resolved_family = _resolve_llama_family(args.model_path, args.tokenizer_path, args.mp_model_family)
    if args.mp_enable_drop and args.mp_max_drop_ratio < 0:
        fam = resolved_family
        if fam == "llama2":
            args.mp_max_drop_ratio = 0.30
        elif fam == "llama3":
            args.mp_max_drop_ratio = 0.40
        else:
            args.mp_max_drop_ratio = 0.50
        print(
            f"Auto-set mp_max_drop_ratio={args.mp_max_drop_ratio:.2f} "
            f"for model_family={fam}"
        )
    print(f"Resolved mp_model_family={resolved_family}")
    if args.mp_quantize_embed_int8 and args.mp_embed_precision != "int8":
        print("Warning: --mp-quantize-embed-int8 is deprecated; overriding --mp-embed-precision to int8.")
        args.mp_embed_precision = "int8"
    if args.mp_quantize_nonsvd_int8 and args.mp_nonsvd_precision != "int8":
        print("Warning: --mp-quantize-nonsvd-int8 is deprecated; overriding --mp-nonsvd-precision to int8.")
        args.mp_nonsvd_precision = "int8"

    def resolve_save_path(save_arg: str) -> str:
        if save_arg is None or len(save_arg.strip()) == 0:
            return save_arg
        save_arg = save_arg.strip()
        is_dir_like = save_arg.endswith(os.sep) or os.path.isdir(save_arg)
        if is_dir_like:
            os.makedirs(save_arg, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            fname = f"svd_mp_ckpt_{stamp}.pt"
            return os.path.join(save_arg, fname)
        parent = os.path.dirname(save_arg)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return save_arg

    def report_checkpoint_size(src_path: str, dst_path: str):
        try:
            dst_bytes = os.path.getsize(dst_path)
        except OSError:
            print(f"Warning: failed to stat saved checkpoint: {dst_path}")
            return
        dst_mib = dst_bytes / (1024.0 * 1024.0)
        msg = f"Saved checkpoint size: {dst_mib:.2f} MiB ({dst_bytes} bytes)"
        if src_path is not None and len(src_path) > 0 and os.path.isfile(src_path):
            try:
                src_bytes = os.path.getsize(src_path)
                src_mib = src_bytes / (1024.0 * 1024.0)
                ratio = float(src_bytes) / max(float(dst_bytes), 1.0)
                msg += f" | source: {src_mib:.2f} MiB | compression_ratio={ratio:.4f}x"
            except OSError:
                pass
        print(msg)

    model, tokenizer = get_model_from_local(args.model_path, tokenizer_path=args.tokenizer_path)
    if args.model_dtype == "fp16":
        model = model.half()
    elif args.model_dtype == "bf16":
        model = model.bfloat16()
    else:
        model = model.float()
    model.eval()
    
    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        tokenizer=tokenizer,
        seqlen=args.model_seq_len,
    )

    pairs = None
    alloc_report = None
    if args.mp_enable:
        if args.mp_enable_drop and args.mp_kfac_mode != "sigma_full":
            raise ValueError("--mp-enable-drop currently requires --mp-kfac-mode sigma_full")
        mp_stages = ["discover_pairs"]
        if args.mp_sigma_mode == "calibrated":
            mp_stages.append("sigma_calibration")
        mp_stages.extend(["kfac_stats", "allocation", "apply_quantization"])
        mp_bar = tqdm(total=len(mp_stages), desc="MP pipeline", dynamic_ncols=True)

        def finish_stage(stage_name: str):
            mp_bar.set_postfix_str(stage_name)
            mp_bar.update(1)

        model = model.to(args.DEV)
        pairs = discover_low_rank_pairs(model)
        if len(pairs) == 0:
            raise RuntimeError("No low-rank *_u_proj/*_v_proj pairs found. Please use an SVD-compressed model.")
        print(f"Found {len(pairs)} low-rank pairs for KFAC-weighted mixed precision. mode={args.mp_kfac_mode}")
        finish_stage("discover_pairs")

        sigma_calib = None
        if args.mp_sigma_mode == "calibrated":
            sigma_calib = calibrate_component_sigma(
                pairs=pairs,
                low_bit=args.mp_low_bit,
                high_bit=args.mp_high_bit,
                explicit_sigma=args.mp_explicit_sigma,
                sigma_eps=args.mp_sigma_eps,
            )
            finish_stage("sigma_calibration")

        if args.mp_kfac_mode == "block_b":
            stats = collect_kfac_stats_block_b(
                model=model,
                pairs=pairs,
                dataloader=dataloader,
                device=args.DEV,
                nsamples=args.mp_kfac_nsamples,
                block_size=args.mp_kfac_block_size,
                collect_a_diag=(args.mp_a_mode != "identity"),
                shrink_lambda=args.mp_b_shrinkage,
                diag_damp=args.mp_b_damp,
                use_grad_checkpointing=args.mp_kfac_grad_checkpointing,
                layerwise=args.mp_kfac_layerwise,
            )
            importance = compute_component_importance_block_b(
                pairs=pairs,
                stats=stats,
                a_mode=args.mp_a_mode,
                adaptive_alpha=None if args.mp_a_alpha < 0 else args.mp_a_alpha,
                adaptive_tau=args.mp_a_adaptive_tau,
            )
            alloc = solve_budgeted_topk(
                pairs=pairs,
                importance=importance,
                low_bit=args.mp_low_bit,
                high_bit=args.mp_high_bit,
                avg_bit=args.mp_avg_bit,
                sigma_calib=sigma_calib,
            )
        elif args.mp_kfac_mode == "diag":
            stats = collect_kfac_stats_diagonal(
                model=model,
                pairs=pairs,
                dataloader=dataloader,
                device=args.DEV,
                nsamples=args.mp_kfac_nsamples,
                use_grad_checkpointing=args.mp_kfac_grad_checkpointing,
                layerwise=args.mp_kfac_layerwise,
            )
            importance = compute_component_importance(pairs, stats)
            alloc = solve_budgeted_topk(
                pairs=pairs,
                importance=importance,
                low_bit=args.mp_low_bit,
                high_bit=args.mp_high_bit,
                avg_bit=args.mp_avg_bit,
                sigma_calib=sigma_calib,
            )
        else:
            whiten_inv = None
            if args.mp_use_whiten:
                if args.mp_whiten_mat_path is None:
                    raise ValueError("--mp-use-whiten requires --mp-whiten-mat-path")
                profiling_mat = torch.load(args.mp_whiten_mat_path, map_location="cpu")
                whiten_inv = build_pair_whiten_inv(
                    pairs=pairs,
                    profiling_mat=profiling_mat,
                    eps=args.mp_whiten_eps,
                )
                print(f"Loaded whitening inverses for {len(whiten_inv)} / {len(pairs)} low-rank pairs")
            stats = collect_kfac_stats_sigma_full(
                model=model,
                pairs=pairs,
                dataloader=dataloader,
                device=args.DEV,
                proj_device=args.mp_kfac_proj_device,
                accum_device=args.mp_kfac_accum_device,
                nsamples=args.mp_kfac_nsamples,
                whiten_inv=whiten_inv,
                use_grad_checkpointing=args.mp_kfac_grad_checkpointing,
                layerwise=args.mp_kfac_layerwise,
                explicit_sigma=args.mp_explicit_sigma,
                sigma_eps=args.mp_sigma_eps,
            )
            fisher_sigma = compute_sigma_fisher_full(pairs, stats)
            if args.mp_enable_drop:
                alloc = solve_budgeted_multilevel_quadratic(
                    pairs=pairs,
                    fisher_sigma=fisher_sigma,
                    low_bit=args.mp_low_bit,
                    high_bit=args.mp_high_bit,
                    avg_bit=args.mp_avg_bit,
                    sigma_calib=sigma_calib,
                    sigma_eps=args.mp_sigma_eps,
                    drop_bit=args.mp_drop_bit,
                    min_keep_ratio=args.mp_min_keep_ratio,
                    max_drop_ratio=args.mp_max_drop_ratio,
                    prefer_low_from_drop=not args.mp_allow_drop_to_high,
                )
            else:
                alloc = solve_budgeted_topk_quadratic(
                    pairs=pairs,
                    fisher_sigma=fisher_sigma,
                    low_bit=args.mp_low_bit,
                    high_bit=args.mp_high_bit,
                    avg_bit=args.mp_avg_bit,
                    sigma_calib=sigma_calib,
                    sigma_eps=args.mp_sigma_eps,
                )
        finish_stage("kfac_stats")
        finish_stage("allocation")

        alloc_report = report_mixed_precision_allocation(
            pairs,
            alloc,
            high_bit=args.mp_high_bit,
            low_bit=args.mp_low_bit,
            drop_bit=args.mp_drop_bit,
            base_bits=base_weight_bits,
            sigma_eps=args.mp_sigma_eps,
        )
        apply_two_path_quantization(
            model=model,
            pairs=pairs,
            alloc=alloc,
            high_bit=args.mp_high_bit,
            low_bit=args.mp_low_bit,
            explicit_sigma=args.mp_explicit_sigma,
            sigma_eps=args.mp_sigma_eps,
            use_int8_kernel=not args.mp_disable_int8_kernel,
            use_int4_kernel=args.mp_enable_int4_kernel,
            int4_quant_type=args.mp_int4_quant_type,
        )
        nonsvd_report = None
        if args.mp_nonsvd_precision == "int8":
            nonsvd_report = apply_non_svd_int8_quantization(
                model=model,
                pairs=pairs,
                use_int8_kernel=not args.mp_disable_int8_kernel,
                exclude_lm_head=args.mp_nonsvd_int8_exclude_lm_head,
            )
            print(
                "Applied non-SVD int8 quantization: "
                f"replaced={nonsvd_report['replaced']}, "
                f"total_linear={nonsvd_report['total_linear']}, "
                f"skipped={nonsvd_report['skipped']}, "
                f"exclude_lm_head={args.mp_nonsvd_int8_exclude_lm_head}"
            )
        else:
            nonsvd_report = apply_non_svd_fp16_cast(
                model=model,
                pairs=pairs,
                exclude_lm_head=args.mp_nonsvd_int8_exclude_lm_head,
            )
            print(
                "Applied non-SVD fp16 cast: "
                f"converted={nonsvd_report['converted']}, "
                f"total_linear={nonsvd_report['total_linear']}, "
                f"skipped={nonsvd_report['skipped']}, "
                f"exclude_lm_head={args.mp_nonsvd_int8_exclude_lm_head}"
            )
        embed_report = None
        if args.mp_embed_precision == "int8":
            embed_report = apply_embed_tokens_int8_quantization(model=model)
            print(
                "Applied embed_tokens int8 quantization: "
                f"replaced={embed_report['replaced']}, "
                f"total_embedding={embed_report['total_embedding']}, "
                f"skipped={embed_report['skipped']}"
            )
        else:
            embed_report = apply_embed_tokens_fp16(model=model)
            print(
                "Applied embed_tokens fp16 cast: "
                f"converted={embed_report['converted']}, "
                f"total_embedding={embed_report['total_embedding']}, "
                f"skipped={embed_report['skipped']}"
            )
        touched = set_mixed_precision_runtime_cache_policy(
            model=model,
            persistent=args.mp_persistent_runtime_cache,
        )
        print(
            f"Configured MP runtime cache policy on {touched} modules: "
            f"persistent={args.mp_persistent_runtime_cache}"
        )
        finish_stage("apply_quantization")
        mp_bar.close()
    elif args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, args.DEV)
        print(time.time() - tick)
    # if args.save:
    #     llama_pack3(model, quantizers)
    #     torch.save(model.state_dict(), args.save)
    if args.save:
        save_path = resolve_save_path(args.save)
        if save_path != args.save:
            print(f"--save points to a directory; resolved checkpoint file: {save_path}")
        if args.mp_enable and pairs is not None:
            strip_stats = strip_original_low_rank_weights(model=model, pairs=pairs)
            print(
                "Stripped original low-rank modules before save: "
                f"removed_params={strip_stats['removed_params']}, "
                f"replaced_modules={strip_stats['replaced_modules']}, "
                f"skipped_pairs={strip_stats['skipped_pairs']}"
            )
        save_format = args.save_format
        if save_format == "auto":
            save_format = "state_dict" if args.mp_enable else "full"

        if save_format == "state_dict":
            cleared = clear_mixed_precision_runtime_cache(model)
            raw_state_dict = model.state_dict()
            packed_state_dict, packed_stats = pack_lowbit_q_state_dict(raw_state_dict)
            payload = {
                "format": "svd_mp_state_dict_v1",
                "state_dict": packed_state_dict,
                "tokenizer_path": args.tokenizer_path,
                "base_model_path": getattr(getattr(model, "config", None), "_name_or_path", None),
                "arch_model_path": args.model_path,
                "model_dtype": args.model_dtype,
                "mp_config": {
                    "enabled": bool(args.mp_enable),
                    "explicit_sigma": bool(args.mp_explicit_sigma),
                    "low_bit": int(args.mp_low_bit),
                    "high_bit": int(args.mp_high_bit),
                    "use_int8_kernel": bool(not args.mp_disable_int8_kernel),
                    "use_int4_kernel": bool(args.mp_enable_int4_kernel),
                    "int4_quant_type": args.mp_int4_quant_type,
                    "quantize_nonsvd_int8": bool(args.mp_nonsvd_precision == "int8"),
                    "nonsvd_precision": args.mp_nonsvd_precision,
                    "nonsvd_int8_exclude_lm_head": bool(args.mp_nonsvd_int8_exclude_lm_head),
                    "quantize_embed_int8": bool(args.mp_embed_precision == "int8"),
                    "embed_precision": args.mp_embed_precision,
                    "target_compression_ratio": float(args.mp_target_compression_ratio),
                    "enable_drop": bool(args.mp_enable_drop),
                    "drop_bit": float(args.mp_drop_bit),
                    "min_keep_ratio": float(args.mp_min_keep_ratio),
                    "max_drop_ratio": float(args.mp_max_drop_ratio),
                    "prefer_low_from_drop": bool(not args.mp_allow_drop_to_high),
                    "model_family": resolved_family,
                    "allocation_report": alloc_report,
                },
                "packed_lowbit_q": True,
                "packed_lowbit_q_stats": packed_stats,
            }
            torch.save(payload, save_path)
            if packed_stats["packed_tensors"] > 0:
                before = packed_stats["before_bytes"] / (1024.0 * 1024.0)
                after = packed_stats["after_bytes"] / (1024.0 * 1024.0)
                ratio = packed_stats["before_bytes"] / max(packed_stats["after_bytes"], 1.0)
                print(
                    "Packed low-bit q tensors: "
                    f"count={int(packed_stats['packed_tensors'])}, "
                    f"bytes_before={before:.2f} MiB, bytes_after={after:.2f} MiB, "
                    f"pack_ratio={ratio:.4f}x"
                )
            print(f"Saved minimal state_dict checkpoint ({cleared} runtime caches cleared) to {save_path}")
            report_checkpoint_size(args.model_path, save_path)
        else:
            torch.save(
                {
                    'model': model,
                    'tokenizer': tokenizer
                },
                save_path,
            )
            print(f"Saved full checkpoint to {save_path}")
            report_checkpoint_size(args.model_path, save_path)

        # Evaluate from reloaded checkpoint to avoid carrying quantization-stage memory/cache.
        print("Reloading saved checkpoint before evaluation to reduce peak memory...")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tokenizer = get_model_from_local(
            save_path,
            tokenizer_path=args.tokenizer_path,
            arch_model_path=args.model_path,
        )
        if args.model_dtype == "fp16":
            model = model.half()
        elif args.model_dtype == "bf16":
            model = model.bfloat16()
        else:
            model = model.float()
        model = model.to(args.DEV)
        model.eval()

    ppl_eval(
        model,
        tokenizer,
        datasets=['wikitext2'],
        model_seq_len=args.model_seq_len,
        batch_size=args.eval_batch_size,
        device=args.DEV,
    )
