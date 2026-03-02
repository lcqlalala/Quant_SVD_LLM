import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PairModules:
    key: str
    stem: str
    parent_path: str
    u_name: str
    v_name: str
    u_module: nn.Module
    v_module: nn.Module


def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    if path == "":
        return model
    cur = model
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _split_parent_and_name(path: str) -> Tuple[str, str]:
    if "." not in path:
        return "", path
    parent, name = path.rsplit(".", 1)
    return parent, name


def _safe_stem(module_name: str) -> Optional[str]:
    if module_name.endswith("_u_proj"):
        return module_name[: -len("_u_proj")]
    if module_name.endswith("_v_proj"):
        return module_name[: -len("_v_proj")]
    return None


def discover_low_rank_pairs(model: nn.Module) -> List[PairModules]:
    named = dict(model.named_modules())
    pairs: List[PairModules] = []
    for path, module in named.items():
        if not isinstance(module, nn.Linear):
            continue
        if not path.endswith("_u_proj"):
            continue
        parent_path, u_name = _split_parent_and_name(path)
        stem = _safe_stem(u_name)
        if stem is None:
            continue
        v_name = f"{stem}_v_proj"
        parent = _get_submodule(model, parent_path)
        if not hasattr(parent, v_name):
            continue
        v_module = getattr(parent, v_name)
        if not isinstance(v_module, nn.Linear):
            continue
        key = f"{parent_path}.{stem}" if parent_path else stem
        pairs.append(
            PairModules(
                key=key,
                stem=stem,
                parent_path=parent_path,
                u_name=u_name,
                v_name=v_name,
                u_module=module,
                v_module=v_module,
            )
        )
    return pairs


def _extract_layer_index(path: str) -> Optional[int]:
    m = re.search(r"\.layers\.(\d+)\.", path)
    if m is None:
        return None
    return int(m.group(1))


def _pair_to_profile_name(pair: PairModules) -> Optional[str]:
    if pair.stem in {"q", "k", "v", "o"}:
        leaf = f"{pair.stem}_proj"
    elif pair.stem == "out":
        leaf = "out_proj"
    elif pair.stem in {"gate", "down", "up"}:
        leaf = f"{pair.stem}_proj"
    elif pair.stem in {"fc1", "fc2"}:
        leaf = pair.stem
    else:
        return None

    if pair.parent_path.endswith(".self_attn") or ".self_attn." in pair.parent_path:
        return f"self_attn.{leaf}"
    if pair.parent_path.endswith(".mlp") or ".mlp." in pair.parent_path:
        return f"mlp.{leaf}"
    if leaf in {"fc1", "fc2"}:
        return leaf
    return None


def build_pair_whiten_inv(
    pairs: List[PairModules],
    profiling_mat: Dict[int, Dict[str, torch.Tensor]],
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Build pair-key -> whitening inverse map from profiling matrices.
    whitening inverse is R^{-1}, where R is Cholesky factor collected during whitening.
    """
    out: Dict[str, torch.Tensor] = {}
    for pair in pairs:
        layer_idx = _extract_layer_index(pair.parent_path)
        if layer_idx is None or layer_idx not in profiling_mat:
            continue
        prof_name = _pair_to_profile_name(pair)
        if prof_name is None:
            continue
        layer_prof = profiling_mat[layer_idx]
        if prof_name not in layer_prof:
            continue
        R = layer_prof[prof_name].float().cpu()
        if eps > 0:
            R = R + eps * torch.eye(R.shape[0], dtype=R.dtype)
        try:
            Rinv = torch.linalg.inv(R)
        except Exception:
            Rinv = torch.linalg.pinv(R)
        out[pair.key] = Rinv
    return out


def _group_pairs_by_layer(pairs: List[PairModules]) -> List[List[PairModules]]:
    groups: Dict[int, List[PairModules]] = {}
    for pair in pairs:
        layer_idx = _extract_layer_index(pair.parent_path)
        key = -1 if layer_idx is None else layer_idx
        groups.setdefault(key, []).append(pair)
    return [groups[k] for k in sorted(groups.keys())]


def _prepare_kfac_mode(model: nn.Module, use_grad_checkpointing: bool):
    use_cache = getattr(model.config, "use_cache", False)
    prev_training = model.training
    model.config.use_cache = False
    gc_enabled = False
    if use_grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.train()
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        gc_enabled = True
    else:
        model.eval()
    return use_cache, prev_training, gc_enabled


def _restore_kfac_mode(model: nn.Module, use_cache: bool, prev_training: bool, gc_enabled: bool):
    model.zero_grad(set_to_none=True)
    if gc_enabled and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = use_cache
    if prev_training:
        model.train()
    else:
        model.eval()


def _compute_sigma_component_norms(
    u_weight: torch.Tensor,
    v_weight: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u_norm = torch.linalg.vector_norm(u_weight.float(), dim=0)
    v_norm = torch.linalg.vector_norm(v_weight.float(), dim=1)
    sigma = u_norm * v_norm
    active = sigma > eps
    u_scale = torch.where(active, u_norm, torch.ones_like(u_norm))
    v_scale = torch.where(active, v_norm, torch.ones_like(v_norm))
    return u_norm, v_norm, sigma, active, u_scale, v_scale


def _decompose_uv_to_explicit_sigma(
    u_weight: torch.Tensor,
    v_weight: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, sigma, active, u_scale, v_scale = _compute_sigma_component_norms(
        u_weight=u_weight,
        v_weight=v_weight,
        eps=eps,
    )
    u_basis = u_weight.float() / u_scale.unsqueeze(0)
    v_basis = v_weight.float() / v_scale.unsqueeze(1)
    if torch.any(~active):
        u_basis[:, ~active] = 0.0
        v_basis[~active, :] = 0.0
        sigma = torch.where(active, sigma, torch.zeros_like(sigma))
    return u_basis, sigma, v_basis, active


def collect_kfac_stats_diagonal(
    model: nn.Module,
    pairs: List[PairModules],
    dataloader,
    device: str = "cuda",
    nsamples: int = 8,
    use_grad_checkpointing: bool = False,
    layerwise: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    use_cache, prev_training, gc_enabled = _prepare_kfac_mode(model, use_grad_checkpointing)

    for pair in pairs:
        in_dim = pair.v_module.in_features
        out_dim = pair.u_module.out_features
        stats[pair.key] = {
            "A_diag": torch.zeros(in_dim, dtype=torch.float64, device=device),
            "B_diag": torch.zeros(out_dim, dtype=torch.float64, device=device),
            "count_a": torch.tensor(0.0, dtype=torch.float64, device=device),
            "count_b": torch.tensor(0.0, dtype=torch.float64, device=device),
        }

        def mk_fwd_pre(k):
            def _hook(module, inputs):
                x = inputs[0].detach()
                x_flat = x.float().reshape(-1, x.shape[-1])
                x2 = x_flat.pow(2).sum(dim=0)
                stats[k]["A_diag"] += x2
                stats[k]["count_a"] += x2.new_tensor(x_flat.shape[0], dtype=torch.float64)

            return _hook

        def mk_bwd(k):
            def _hook(module, grad_input, grad_output):
                g = grad_output[0].detach()
                g_flat = g.float().reshape(-1, g.shape[-1])
                g2 = g_flat.pow(2).sum(dim=0)
                stats[k]["B_diag"] += g2
                stats[k]["count_b"] += g2.new_tensor(g_flat.shape[0], dtype=torch.float64)

            return _hook

    pair_subsets = _group_pairs_by_layer(pairs) if layerwise else [pairs]
    for subset in pair_subsets:
        handles = []
        for pair in subset:
            handles.append(pair.v_module.register_forward_pre_hook(mk_fwd_pre(pair.key)))
            handles.append(pair.u_module.register_full_backward_hook(mk_bwd(pair.key)))

        for idx, batch in enumerate(dataloader):
            if idx >= nsamples:
                break
            input_ids, labels = batch[0].to(device), batch[1].to(device)
            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            out.loss.backward()

        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    for key, v in stats.items():
        ca = max(v["count_a"].item(), 1.0)
        cb = max(v["count_b"].item(), 1.0)
        v["A_diag"] = (v["A_diag"] / ca).float().cpu()
        v["B_diag"] = (v["B_diag"] / cb).float().cpu()
        del v["count_a"]
        del v["count_b"]

    _restore_kfac_mode(model, use_cache, prev_training, gc_enabled)
    return stats


def collect_kfac_stats_block_b(
    model: nn.Module,
    pairs: List[PairModules],
    dataloader,
    device: str = "cuda",
    nsamples: int = 8,
    block_size: int = 128,
    collect_a_diag: bool = True,
    shrink_lambda: float = 0.1,
    diag_damp: float = 1e-6,
    use_grad_checkpointing: bool = False,
    layerwise: bool = False,
) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    use_cache, prev_training, gc_enabled = _prepare_kfac_mode(model, use_grad_checkpointing)

    for pair in pairs:
        out_dim = pair.u_module.out_features
        nblocks = (out_dim + block_size - 1) // block_size
        stats[pair.key] = {
            "block_size": block_size,
            "out_dim": out_dim,
            "B_blocks": [],
            "count_b": 0.0,
            "A_diag": torch.zeros(pair.v_module.in_features, dtype=torch.float64),
            "count_a": 0.0,
        }
        for bi in range(nblocks):
            s = bi * block_size
            e = min((bi + 1) * block_size, out_dim)
            stats[pair.key]["B_blocks"].append(torch.zeros((e - s, e - s), dtype=torch.float64))

        def mk_fwd_pre(k):
            def _hook(module, inputs):
                if not collect_a_diag:
                    return
                x = inputs[0].detach().float()
                x_flat = x.reshape(-1, x.shape[-1])
                stats[k]["A_diag"] += x_flat.pow(2).sum(dim=0).to(dtype=torch.float64).cpu()
                stats[k]["count_a"] += float(x_flat.shape[0])

            return _hook

        def mk_bwd(k):
            def _hook(module, grad_input, grad_output):
                g = grad_output[0].detach().float()
                g_flat = g.reshape(-1, g.shape[-1])
                stats[k]["count_b"] += float(g_flat.shape[0])
                bsz = int(stats[k]["block_size"])
                for bi, blk in enumerate(stats[k]["B_blocks"]):
                    s = bi * bsz
                    e = min((bi + 1) * bsz, g_flat.shape[-1])
                    gb = g_flat[:, s:e]
                    cov = gb.transpose(0, 1).matmul(gb).to(dtype=torch.float64).cpu()
                    stats[k]["B_blocks"][bi].add_(cov)

            return _hook

    pair_subsets = _group_pairs_by_layer(pairs) if layerwise else [pairs]
    for subset in pair_subsets:
        handles = []
        for pair in subset:
            handles.append(pair.v_module.register_forward_pre_hook(mk_fwd_pre(pair.key)))
            handles.append(pair.u_module.register_full_backward_hook(mk_bwd(pair.key)))

        for idx, batch in enumerate(dataloader):
            if idx >= nsamples:
                break
            input_ids, labels = batch[0].to(device), batch[1].to(device)
            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            out.loss.backward()

        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    for key, v in stats.items():
        cb = max(float(v["count_b"]), 1.0)
        for bi in range(len(v["B_blocks"])):
            blk = (v["B_blocks"][bi] / cb).float()
            if shrink_lambda > 0:
                blk_diag = torch.diag(torch.diag(blk))
                blk = (1.0 - shrink_lambda) * blk + shrink_lambda * blk_diag
            if diag_damp > 0:
                blk = blk + diag_damp * torch.eye(blk.shape[0], dtype=blk.dtype)
            v["B_blocks"][bi] = blk
        ca = max(float(v["count_a"]), 1.0)
        if collect_a_diag:
            v["A_diag"] = (v["A_diag"] / ca).float()
        else:
            v["A_diag"] = None
        del v["count_b"]
        del v["count_a"]

    _restore_kfac_mode(model, use_cache, prev_training, gc_enabled)
    return stats


def collect_kfac_stats_sigma_full(
    model: nn.Module,
    pairs: List[PairModules],
    dataloader,
    device: str = "cuda",
    nsamples: int = 8,
    whiten_inv: Optional[Dict[str, torch.Tensor]] = None,
    use_grad_checkpointing: bool = False,
    layerwise: bool = False,
    explicit_sigma: bool = False,
    sigma_eps: float = 1e-12,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collect K-FAC moments projected to rank-space bases:
      G_sigma = U^T G U, A_sigma = V A V^T.
    These are sufficient to build full sigma-space Fisher:
      F_sigma = G_sigma * A_sigma (Hadamard product).
    """
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    use_cache, prev_training, gc_enabled = _prepare_kfac_mode(model, use_grad_checkpointing)
    explicit_meta: Dict[str, Dict[str, torch.Tensor]] = {}

    for pair in pairs:
        rank = pair.u_module.weight.shape[1]
        stats[pair.key] = {
            "A_proj": torch.zeros((rank, rank), dtype=torch.float64),
            "G_proj": torch.zeros((rank, rank), dtype=torch.float64),
            "count_a": torch.tensor(0.0, dtype=torch.float64),
            "count_g": torch.tensor(0.0, dtype=torch.float64),
        }
        if explicit_sigma:
            _, _, _, active, u_scale, v_scale = _compute_sigma_component_norms(
                u_weight=pair.u_module.weight.detach(),
                v_weight=pair.v_module.weight.detach(),
                eps=sigma_eps,
            )
            explicit_meta[pair.key] = {
                "active": active.cpu(),
                "u_scale": u_scale.cpu(),
                "v_scale": v_scale.cpu(),
            }

        def mk_fwd_pre(k):
            def _hook(module, inputs):
                x = inputs[0].detach().float()
                x_flat = x.reshape(-1, x.shape[-1])
                if whiten_inv is not None and k in whiten_inv:
                    x_flat = x_flat.matmul(whiten_inv[k].to(device=x_flat.device, dtype=x_flat.dtype))
                # module is *_v_proj with shape [rank, in_dim]
                V = module.weight.detach().float()
                z = x_flat.matmul(V.transpose(0, 1))
                if explicit_sigma and k in explicit_meta:
                    v_scale = explicit_meta[k]["v_scale"].to(device=z.device, dtype=z.dtype)
                    active = explicit_meta[k]["active"].to(device=z.device, dtype=z.dtype)
                    z = z / v_scale.unsqueeze(0)
                    z = z * active.unsqueeze(0)
                stats[k]["A_proj"] += z.transpose(0, 1).matmul(z).to(dtype=torch.float64).cpu()
                stats[k]["count_a"] += z.new_tensor(float(z.shape[0]), dtype=torch.float64).cpu()

            return _hook

        def mk_bwd(k):
            def _hook(module, grad_input, grad_output):
                g = grad_output[0].detach().float()
                g_flat = g.reshape(-1, g.shape[-1])
                # module is *_u_proj with shape [out_dim, rank]
                U = module.weight.detach().float()
                s = g_flat.matmul(U)
                if explicit_sigma and k in explicit_meta:
                    u_scale = explicit_meta[k]["u_scale"].to(device=s.device, dtype=s.dtype)
                    active = explicit_meta[k]["active"].to(device=s.device, dtype=s.dtype)
                    s = s / u_scale.unsqueeze(0)
                    s = s * active.unsqueeze(0)
                stats[k]["G_proj"] += s.transpose(0, 1).matmul(s).to(dtype=torch.float64).cpu()
                stats[k]["count_g"] += s.new_tensor(float(s.shape[0]), dtype=torch.float64).cpu()

            return _hook

    pair_subsets = _group_pairs_by_layer(pairs) if layerwise else [pairs]
    for subset in pair_subsets:
        handles = []
        for pair in subset:
            handles.append(pair.v_module.register_forward_pre_hook(mk_fwd_pre(pair.key)))
            handles.append(pair.u_module.register_full_backward_hook(mk_bwd(pair.key)))

        for idx, batch in enumerate(dataloader):
            if idx >= nsamples:
                break
            input_ids, labels = batch[0].to(device), batch[1].to(device)
            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            out.loss.backward()

        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    for key, v in stats.items():
        ca = max(v["count_a"].item(), 1.0)
        cg = max(v["count_g"].item(), 1.0)
        v["A_proj"] = (v["A_proj"] / ca).float().cpu()
        v["G_proj"] = (v["G_proj"] / cg).float().cpu()
        del v["count_a"]
        del v["count_g"]

    _restore_kfac_mode(model, use_cache, prev_training, gc_enabled)
    return stats


def compute_sigma_fisher_full(
    pairs: List[PairModules],
    stats: Dict[str, Dict[str, torch.Tensor]],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for pair in pairs:
        if pair.key not in stats:
            continue
        A_proj = stats[pair.key]["A_proj"].float().cpu()
        G_proj = stats[pair.key]["G_proj"].float().cpu()
        F_sigma = G_proj * A_proj
        F_sigma = 0.5 * (F_sigma + F_sigma.transpose(0, 1))
        if eps > 0:
            F_sigma = F_sigma + eps * torch.eye(F_sigma.shape[0], dtype=F_sigma.dtype)
        out[pair.key] = F_sigma
    return out


def compute_component_importance(
    pairs: List[PairModules], stats: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for pair in pairs:
        if pair.key not in stats:
            continue
        A_diag = stats[pair.key]["A_diag"].to(pair.v_module.weight.device, dtype=torch.float32)
        B_diag = stats[pair.key]["B_diag"].to(pair.u_module.weight.device, dtype=torch.float32)
        U = pair.u_module.weight.data.float()
        V = pair.v_module.weight.data.float()
        u_term = (U.pow(2) * B_diag[:, None]).sum(dim=0)
        v_term = (V.pow(2) * A_diag[None, :]).sum(dim=1)
        score = (u_term * v_term).clamp_min(1e-12)
        out[pair.key] = score.detach().cpu()
    return out


def compute_component_importance_block_b(
    pairs: List[PairModules],
    stats: Dict[str, Dict[str, object]],
    a_mode: str = "adaptive",
    adaptive_alpha: Optional[float] = None,
    adaptive_tau: float = 0.5,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for pair in pairs:
        if pair.key not in stats:
            continue
        U = pair.u_module.weight.data.float()
        V = pair.v_module.weight.data.float()
        dev = U.device
        rank = U.shape[1]
        b_term = torch.zeros(rank, device=dev, dtype=torch.float32)
        blk_size = int(stats[pair.key]["block_size"])
        B_blocks: List[torch.Tensor] = stats[pair.key]["B_blocks"]  # type: ignore[assignment]
        for bi, B_blk_cpu in enumerate(B_blocks):
            s = bi * blk_size
            e = min((bi + 1) * blk_size, U.shape[0])
            U_blk = U[s:e, :]
            B_blk = B_blk_cpu.to(device=dev, dtype=torch.float32)
            b_term += (U_blk * (B_blk.matmul(U_blk))).sum(dim=0)

        if a_mode == "identity":
            a_term = V.pow(2).sum(dim=1)
        else:
            A_diag_cpu = stats[pair.key].get("A_diag", None)
            if A_diag_cpu is None:
                raise ValueError("A_diag was not collected but a_mode is not identity")
            A_diag = A_diag_cpu.to(dev, dtype=torch.float32)
            if a_mode == "diag":
                a_weight = A_diag
            elif a_mode == "adaptive":
                mean_a = A_diag.mean().clamp_min(1e-12)
                norm_a = A_diag / mean_a
                if adaptive_alpha is None:
                    cv2 = torch.mean((norm_a - 1.0).pow(2))
                    alpha = (cv2 / (cv2 + adaptive_tau)).clamp(0.0, 1.0)
                else:
                    alpha = torch.tensor(float(adaptive_alpha), device=dev, dtype=torch.float32).clamp(0.0, 1.0)
                a_weight = (1.0 - alpha) + alpha * norm_a
            else:
                raise ValueError(f"Unsupported a_mode: {a_mode}")
            a_term = (V.pow(2) * a_weight[None, :]).sum(dim=1)
        score = (b_term * a_term).clamp_min(1e-12)
        out[pair.key] = score.detach().cpu()
    return out


def _quant_noise_proxy(bits: int) -> float:
    if bits <= 1:
        return 1.0
    q = (2 ** (bits - 1)) - 1
    return 1.0 / float(q * q)


def solve_budgeted_topk(
    pairs: List[PairModules],
    importance: Dict[str, torch.Tensor],
    low_bit: int = 4,
    high_bit: int = 8,
    avg_bit: float = 4.5,
    sigma_calib: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    if high_bit <= low_bit:
        raise ValueError("high_bit must be larger than low_bit")

    items = []
    total_params = 0.0
    for pair in pairs:
        score = importance.get(pair.key, None)
        if score is None:
            continue
        out_dim, rank = pair.u_module.weight.shape
        _, in_dim = pair.v_module.weight.shape
        params_per_component = float(out_dim + in_dim)
        total_params += params_per_component * float(rank)
        for i in range(rank):
            if sigma_calib is not None and pair.key in sigma_calib:
                sigma_low = float(sigma_calib[pair.key]["low"][i].item())
                sigma_high = float(sigma_calib[pair.key]["high"][i].item())
                gain = max(sigma_low - sigma_high, 0.0)
            else:
                gain = _quant_noise_proxy(low_bit) - _quant_noise_proxy(high_bit)
            value = float(score[i].item()) * gain
            delta_cost = params_per_component * float(high_bit - low_bit)
            density = value / max(delta_cost, 1e-12)
            items.append((density, value, delta_cost, pair.key, i))

    target_avg = min(max(avg_bit, float(low_bit)), float(high_bit))
    extra_budget = total_params * (target_avg - float(low_bit))

    items.sort(key=lambda x: x[0], reverse=True)
    selected = set()
    used = 0.0
    for _, value, delta_cost, key, idx in items:
        if value <= 0:
            continue
        if used + delta_cost <= extra_budget + 1e-9:
            selected.add((key, idx))
            used += delta_cost

    alloc: Dict[str, torch.Tensor] = {}
    for pair in pairs:
        rank = pair.u_module.weight.shape[1]
        mask = torch.zeros(rank, dtype=torch.bool)
        for i in range(rank):
            if (pair.key, i) in selected:
                mask[i] = True
        alloc[pair.key] = mask
    return alloc


def _estimate_sigma_proxy(pair: PairModules) -> torch.Tensor:
    """
    Rank-component magnitude proxy in low-rank chain W = U V.
    For component i, ||u_i v_i^T||_F = ||u_i|| * ||v_i||.
    """
    _, _, sigma, _, _, _ = _compute_sigma_component_norms(
        u_weight=pair.u_module.weight.data,
        v_weight=pair.v_module.weight.data,
        eps=0.0,
    )
    return sigma.cpu()


def solve_budgeted_topk_quadratic(
    pairs: List[PairModules],
    fisher_sigma: Dict[str, torch.Tensor],
    low_bit: int = 4,
    high_bit: int = 8,
    avg_bit: float = 4.5,
    sigma_calib: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    sigma_eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Budgeted greedy selection with sigma-space quadratic objective:
      L ~ 1/2 * z^T F_sigma z
    where z is component perturbation scale.
    Start from all-low-bit baseline, then upgrade components to high-bit.
    """
    if high_bit <= low_bit:
        raise ValueError("high_bit must be larger than low_bit")

    total_params = 0.0
    states: Dict[str, Dict[str, object]] = {}
    alloc: Dict[str, torch.Tensor] = {}
    noise_low = math.sqrt(_quant_noise_proxy(low_bit))
    noise_high = math.sqrt(_quant_noise_proxy(high_bit))

    for pair in pairs:
        out_dim, rank = pair.u_module.weight.shape
        _, in_dim = pair.v_module.weight.shape
        params_per_component = float(out_dim + in_dim)
        alloc[pair.key] = torch.zeros(rank, dtype=torch.bool)

        if pair.key not in fisher_sigma:
            continue
        F = fisher_sigma[pair.key].float().cpu()
        if F.shape != (rank, rank):
            continue

        sigma_vec = _estimate_sigma_proxy(pair).float().cpu()
        if sigma_calib is not None and pair.key in sigma_calib:
            low_vec = sigma_calib[pair.key]["low"].float().clamp_min(0.0).sqrt().cpu()
            high_vec = sigma_calib[pair.key]["high"].float().clamp_min(0.0).sqrt().cpu()
        else:
            low_vec = torch.full((rank,), noise_low, dtype=torch.float32)
            high_vec = torch.full((rank,), noise_high, dtype=torch.float32)

        active = sigma_vec > sigma_eps
        active_count = int(active.sum().item())
        if active_count == 0:
            continue
        total_params += params_per_component * float(active_count)

        z = sigma_vec * low_vec
        delta = sigma_vec * (high_vec - low_vec)  # negative when high<low
        F = 0.5 * (F + F.transpose(0, 1))
        Fz = F.matmul(z)

        states[pair.key] = {
            "F": F,
            "Fz": Fz,
            "delta": delta,
            "mask": torch.zeros(rank, dtype=torch.bool),
            "active": active,
            "diag": torch.diag(F),
            "cost": params_per_component * float(high_bit - low_bit),
        }

    target_avg = min(max(avg_bit, float(low_bit)), float(high_bit))
    extra_budget = total_params * (target_avg - float(low_bit))
    used = 0.0

    while True:
        best_key = None
        best_idx = -1
        best_density = 0.0
        best_gain = 0.0
        best_cost = 0.0

        for key, st in states.items():
            cost = float(st["cost"])
            if used + cost > extra_budget + 1e-9:
                continue
            mask = st["mask"]  # type: ignore[assignment]
            active = st["active"]  # type: ignore[assignment]
            Fz = st["Fz"]  # type: ignore[assignment]
            delta = st["delta"]  # type: ignore[assignment]
            diag = st["diag"]  # type: ignore[assignment]
            remain = (~mask) & active
            if not torch.any(remain):
                continue
            cand_idx = torch.where(remain)[0]
            # gain = L(old)-L(new) for one-coordinate update z_i += delta_i
            cand_delta = delta[cand_idx]
            cand_gain = -(cand_delta * Fz[cand_idx] + 0.5 * (cand_delta * cand_delta) * diag[cand_idx])
            max_gain, pos = torch.max(cand_gain, dim=0)
            gain = float(max_gain.item())
            if gain <= 0.0:
                continue
            density = gain / max(cost, 1e-12)
            if density > best_density:
                best_density = density
                best_key = key
                best_idx = int(cand_idx[int(pos.item())].item())
                best_gain = gain
                best_cost = cost

        if best_key is None or best_idx < 0 or best_gain <= 0.0:
            break

        st = states[best_key]
        delta_i = float(st["delta"][best_idx].item())  # type: ignore[index]
        st["mask"][best_idx] = True  # type: ignore[index]
        st["Fz"] += st["F"][:, best_idx] * delta_i  # type: ignore[index]
        used += best_cost

    for key, st in states.items():
        alloc[key] = st["mask"].clone()  # type: ignore[assignment]
    return alloc


def _quantize_vec_symmetric(vec: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = float((2 ** (bits - 1)) - 1)
    if qmax <= 0:
        return torch.zeros_like(vec)
    scale = vec.abs().max().clamp_min(1e-8) / qmax
    q = torch.round(vec / scale).clamp(-qmax, qmax)
    return q * scale


def _rank1_outer_error(u: torch.Tensor, v: torch.Tensor, uq: torch.Tensor, vq: torch.Tensor) -> torch.Tensor:
    uu = torch.dot(u, u)
    vv = torch.dot(v, v)
    uquq = torch.dot(uq, uq)
    vqvq = torch.dot(vq, vq)
    cross = torch.dot(u, uq) * torch.dot(v, vq)
    return (uu * vv + uquq * vqvq - 2.0 * cross).clamp_min(0.0)


def calibrate_component_sigma(
    pairs: List[PairModules],
    low_bit: int = 4,
    high_bit: int = 8,
    explicit_sigma: bool = False,
    sigma_eps: float = 1e-12,
) -> Dict[str, Dict[str, torch.Tensor]]:
    sigma: Dict[str, Dict[str, torch.Tensor]] = {}
    for pair in pairs:
        U = pair.u_module.weight.data.float().cpu()
        V = pair.v_module.weight.data.float().cpu()
        U_basis = None
        V_basis = None
        sigma_vec = None
        active = None
        if explicit_sigma:
            U_basis, sigma_vec, V_basis, active = _decompose_uv_to_explicit_sigma(
                u_weight=U,
                v_weight=V,
                eps=sigma_eps,
            )
        rank = U.shape[1]
        low = torch.zeros(rank, dtype=torch.float32)
        high = torch.zeros(rank, dtype=torch.float32)
        for i in range(rank):
            if explicit_sigma:
                if active is None or sigma_vec is None or U_basis is None or V_basis is None:
                    continue
                if not bool(active[i].item()):
                    continue
                root_sigma = math.sqrt(max(float(sigma_vec[i].item()), 0.0))
                u = U_basis[:, i] * root_sigma
                v = V_basis[i, :] * root_sigma
            else:
                u = U[:, i]
                v = V[i, :]
            uq_l = _quantize_vec_symmetric(u, low_bit)
            vq_l = _quantize_vec_symmetric(v, low_bit)
            uq_h = _quantize_vec_symmetric(u, high_bit)
            vq_h = _quantize_vec_symmetric(v, high_bit)
            low[i] = _rank1_outer_error(u, v, uq_l, vq_l)
            high[i] = _rank1_outer_error(u, v, uq_h, vq_h)
        sigma[pair.key] = {"low": low, "high": high}
    return sigma


class TwoPathLowRankLinear(nn.Module):
    def __init__(
        self,
        u_weight: torch.Tensor,
        v_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        high_idx: torch.Tensor,
        low_idx: torch.Tensor,
        high_bit: int = 8,
        low_bit: int = 4,
    ):
        super().__init__()
        self.high_bit = high_bit
        self.low_bit = low_bit
        self.out_features = u_weight.shape[0]
        self._runtime_u: Optional[torch.Tensor] = None
        self._runtime_v: Optional[torch.Tensor] = None
        self._runtime_device: Optional[torch.device] = None
        self._runtime_dtype: Optional[torch.dtype] = None
        self.register_buffer("high_idx", high_idx.to(torch.long), persistent=False)
        self.register_buffer("low_idx", low_idx.to(torch.long), persistent=False)

        if self.high_idx.numel() > 0:
            uh, vh = u_weight[:, self.high_idx], v_weight[self.high_idx, :]
            self.register_buffer("uh_q", self._q_per_row(uh, high_bit)[0], persistent=True)
            self.register_buffer("uh_s", self._q_per_row(uh, high_bit)[1], persistent=True)
            self.register_buffer("vh_q", self._q_per_row(vh, high_bit)[0], persistent=True)
            self.register_buffer("vh_s", self._q_per_row(vh, high_bit)[1], persistent=True)
        else:
            self.register_buffer("uh_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("uh_s", torch.empty(0), persistent=True)
            self.register_buffer("vh_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("vh_s", torch.empty(0), persistent=True)

        if self.low_idx.numel() > 0:
            ul, vl = u_weight[:, self.low_idx], v_weight[self.low_idx, :]
            self.register_buffer("ul_q", self._q_per_row(ul, low_bit)[0], persistent=True)
            self.register_buffer("ul_s", self._q_per_row(ul, low_bit)[1], persistent=True)
            self.register_buffer("vl_q", self._q_per_row(vl, low_bit)[0], persistent=True)
            self.register_buffer("vl_s", self._q_per_row(vl, low_bit)[1], persistent=True)
        else:
            self.register_buffer("ul_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("ul_s", torch.empty(0), persistent=True)
            self.register_buffer("vl_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("vl_s", torch.empty(0), persistent=True)

        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone().float(), persistent=True)

    @staticmethod
    def _q_per_row(w: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        qmax = float((2 ** (bits - 1)) - 1)
        w = w.float()
        s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(w / s).clamp(-qmax, qmax).to(torch.int8)
        return q.cpu(), s.squeeze(1).cpu()

    @staticmethod
    def _deq_per_row(q: torch.Tensor, s: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if q.numel() == 0:
            return torch.empty(0, device=device, dtype=dtype)
        qf = q.to(device=device, dtype=torch.float32)
        sf = s.to(device=device, dtype=torch.float32).unsqueeze(1)
        return (qf * sf).to(dtype=dtype)

    def _build_runtime_cache(self, device: torch.device, dtype: torch.dtype):
        if (
            self._runtime_u is not None
            and self._runtime_v is not None
            and self._runtime_device == device
            and self._runtime_dtype == dtype
        ):
            return
        parts_u = []
        parts_v = []
        if self.uh_q.numel() > 0:
            parts_u.append(self._deq_per_row(self.uh_q, self.uh_s, device, dtype))
            parts_v.append(self._deq_per_row(self.vh_q, self.vh_s, device, dtype))
        if self.ul_q.numel() > 0:
            parts_u.append(self._deq_per_row(self.ul_q, self.ul_s, device, dtype))
            parts_v.append(self._deq_per_row(self.vl_q, self.vl_s, device, dtype))
        if len(parts_u) == 0:
            self._runtime_u = torch.zeros((self.out_features, 0), device=device, dtype=dtype)
            self._runtime_v = torch.zeros((0, 0), device=device, dtype=dtype)
        elif len(parts_u) == 1:
            self._runtime_u = parts_u[0]
            self._runtime_v = parts_v[0]
        else:
            # Runtime fuse two paths into one low-rank chain: y = [Uh Ul]([Vh;Vl]x)
            self._runtime_u = torch.cat(parts_u, dim=1)
            self._runtime_v = torch.cat(parts_v, dim=0)
        self._runtime_device = device
        self._runtime_dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        dev = x.device
        self._build_runtime_cache(dev, dtype)
        if self._runtime_u is None or self._runtime_v is None or self._runtime_v.numel() == 0:
            out = torch.zeros((*x.shape[:-1], self.out_features), device=dev, dtype=dtype)
        else:
            z = F.linear(x, self._runtime_v)
            out = F.linear(z, self._runtime_u)
        if self.bias is not None:
            out = out + self.bias.to(device=dev, dtype=dtype)
        return out


class TwoPathSigmaLowRankLinear(nn.Module):
    def __init__(
        self,
        u_basis: torch.Tensor,
        v_basis: torch.Tensor,
        sigma: torch.Tensor,
        bias: Optional[torch.Tensor],
        high_idx: torch.Tensor,
        low_idx: torch.Tensor,
        high_bit: int = 8,
        low_bit: int = 4,
    ):
        super().__init__()
        self.high_bit = high_bit
        self.low_bit = low_bit
        self.out_features = u_basis.shape[0]
        self.in_features = v_basis.shape[1]
        self._runtime_u: Optional[torch.Tensor] = None
        self._runtime_v: Optional[torch.Tensor] = None
        self._runtime_sigma: Optional[torch.Tensor] = None
        self._runtime_device: Optional[torch.device] = None
        self._runtime_dtype: Optional[torch.dtype] = None
        self.register_buffer("high_idx", high_idx.to(torch.long), persistent=False)
        self.register_buffer("low_idx", low_idx.to(torch.long), persistent=False)

        if self.high_idx.numel() > 0:
            uh = u_basis[:, self.high_idx]
            vh = v_basis[self.high_idx, :]
            sh = sigma[self.high_idx].float().cpu()
            self.register_buffer("uh_q", self._q_per_row(uh, high_bit)[0], persistent=True)
            self.register_buffer("uh_s", self._q_per_row(uh, high_bit)[1], persistent=True)
            self.register_buffer("vh_q", self._q_per_row(vh, high_bit)[0], persistent=True)
            self.register_buffer("vh_s", self._q_per_row(vh, high_bit)[1], persistent=True)
            self.register_buffer("sh", sh, persistent=True)
        else:
            self.register_buffer("uh_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("uh_s", torch.empty(0), persistent=True)
            self.register_buffer("vh_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("vh_s", torch.empty(0), persistent=True)
            self.register_buffer("sh", torch.empty(0), persistent=True)

        if self.low_idx.numel() > 0:
            ul = u_basis[:, self.low_idx]
            vl = v_basis[self.low_idx, :]
            sl = sigma[self.low_idx].float().cpu()
            self.register_buffer("ul_q", self._q_per_row(ul, low_bit)[0], persistent=True)
            self.register_buffer("ul_s", self._q_per_row(ul, low_bit)[1], persistent=True)
            self.register_buffer("vl_q", self._q_per_row(vl, low_bit)[0], persistent=True)
            self.register_buffer("vl_s", self._q_per_row(vl, low_bit)[1], persistent=True)
            self.register_buffer("sl", sl, persistent=True)
        else:
            self.register_buffer("ul_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("ul_s", torch.empty(0), persistent=True)
            self.register_buffer("vl_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("vl_s", torch.empty(0), persistent=True)
            self.register_buffer("sl", torch.empty(0), persistent=True)

        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone().float(), persistent=True)

    @staticmethod
    def _q_per_row(w: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        qmax = float((2 ** (bits - 1)) - 1)
        w = w.float()
        s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(w / s).clamp(-qmax, qmax).to(torch.int8)
        return q.cpu(), s.squeeze(1).cpu()

    @staticmethod
    def _deq_per_row(q: torch.Tensor, s: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if q.numel() == 0:
            return torch.empty(0, device=device, dtype=dtype)
        qf = q.to(device=device, dtype=torch.float32)
        sf = s.to(device=device, dtype=torch.float32).unsqueeze(1)
        return (qf * sf).to(dtype=dtype)

    def _build_runtime_cache(self, device: torch.device, dtype: torch.dtype):
        if (
            self._runtime_u is not None
            and self._runtime_v is not None
            and self._runtime_sigma is not None
            and self._runtime_device == device
            and self._runtime_dtype == dtype
        ):
            return
        parts_u = []
        parts_v = []
        parts_sigma = []
        if self.uh_q.numel() > 0:
            parts_u.append(self._deq_per_row(self.uh_q, self.uh_s, device, dtype))
            parts_v.append(self._deq_per_row(self.vh_q, self.vh_s, device, dtype))
            parts_sigma.append(self.sh.to(device=device, dtype=dtype))
        if self.ul_q.numel() > 0:
            parts_u.append(self._deq_per_row(self.ul_q, self.ul_s, device, dtype))
            parts_v.append(self._deq_per_row(self.vl_q, self.vl_s, device, dtype))
            parts_sigma.append(self.sl.to(device=device, dtype=dtype))
        if len(parts_u) == 0:
            self._runtime_u = torch.zeros((self.out_features, 0), device=device, dtype=dtype)
            self._runtime_v = torch.zeros((0, self.in_features), device=device, dtype=dtype)
            self._runtime_sigma = torch.zeros((0,), device=device, dtype=dtype)
        elif len(parts_u) == 1:
            self._runtime_u = parts_u[0]
            self._runtime_v = parts_v[0]
            self._runtime_sigma = parts_sigma[0]
        else:
            self._runtime_u = torch.cat(parts_u, dim=1)
            self._runtime_v = torch.cat(parts_v, dim=0)
            self._runtime_sigma = torch.cat(parts_sigma, dim=0)
        self._runtime_device = device
        self._runtime_dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        dev = x.device
        self._build_runtime_cache(dev, dtype)
        if (
            self._runtime_u is None
            or self._runtime_v is None
            or self._runtime_sigma is None
            or self._runtime_v.numel() == 0
        ):
            out = torch.zeros((*x.shape[:-1], self.out_features), device=dev, dtype=dtype)
        else:
            z = F.linear(x, self._runtime_v)
            z = z * self._runtime_sigma
            out = F.linear(z, self._runtime_u)
        if self.bias is not None:
            out = out + self.bias.to(device=dev, dtype=dtype)
        return out


def apply_two_path_quantization(
    model: nn.Module,
    pairs: List[PairModules],
    alloc: Dict[str, torch.Tensor],
    high_bit: int = 8,
    low_bit: int = 4,
    explicit_sigma: bool = False,
    sigma_eps: float = 1e-12,
):
    for pair in pairs:
        if pair.key not in alloc:
            continue
        high_mask = alloc[pair.key]
        rank = high_mask.numel()
        high_idx = torch.where(high_mask)[0]
        low_idx = torch.where(~high_mask)[0]
        if high_idx.numel() == 0 and low_idx.numel() == 0 and rank > 0:
            low_idx = torch.arange(rank)

        if explicit_sigma:
            u_basis, sigma_vec, v_basis, _ = _decompose_uv_to_explicit_sigma(
                u_weight=pair.u_module.weight.data.detach().float().cpu(),
                v_weight=pair.v_module.weight.data.detach().float().cpu(),
                eps=sigma_eps,
            )
            mp = TwoPathSigmaLowRankLinear(
                u_basis=u_basis,
                v_basis=v_basis,
                sigma=sigma_vec,
                bias=pair.u_module.bias.data if pair.u_module.bias is not None else None,
                high_idx=high_idx,
                low_idx=low_idx,
                high_bit=high_bit,
                low_bit=low_bit,
            )
        else:
            mp = TwoPathLowRankLinear(
                u_weight=pair.u_module.weight.data,
                v_weight=pair.v_module.weight.data,
                bias=pair.u_module.bias.data if pair.u_module.bias is not None else None,
                high_idx=high_idx,
                low_idx=low_idx,
                high_bit=high_bit,
                low_bit=low_bit,
            )
        parent = _get_submodule(model, pair.parent_path)
        setattr(parent, f"{pair.stem}_mp_proj", mp)
