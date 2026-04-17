import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    import bitsandbytes as bnb  # type: ignore[import-not-found]
except Exception:
    bnb = None

MP_STATE_DROP = 0
MP_STATE_LOW = 1
MP_STATE_HIGH = 2


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
    accum_device = torch.device(device)

    for pair in pairs:
        out_dim = pair.u_module.out_features
        nblocks = (out_dim + block_size - 1) // block_size
        stats[pair.key] = {
            "block_size": block_size,
            "out_dim": out_dim,
            "B_blocks": [],
            "count_b": 0.0,
            "A_diag": torch.zeros(pair.v_module.in_features, dtype=torch.float32, device=accum_device),
            "count_a": 0.0,
        }
        for bi in range(nblocks):
            s = bi * block_size
            e = min((bi + 1) * block_size, out_dim)
            stats[pair.key]["B_blocks"].append(torch.zeros((e - s, e - s), dtype=torch.float32, device=accum_device))

        def mk_fwd_pre(k):
            def _hook(module, inputs):
                if not collect_a_diag:
                    return
                x = inputs[0].detach().float()
                x_flat = x.reshape(-1, x.shape[-1])
                stats[k]["A_diag"] += x_flat.pow(2).sum(dim=0)
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
                    cov = gb.transpose(0, 1).matmul(gb).to(dtype=torch.float32)
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

    for key, v in stats.items():
        cb = max(float(v["count_b"]), 1.0)
        for bi in range(len(v["B_blocks"])):
            blk = (v["B_blocks"][bi] / cb).float()
            if shrink_lambda > 0:
                blk_diag = torch.diag(torch.diag(blk))
                blk = (1.0 - shrink_lambda) * blk + shrink_lambda * blk_diag
            if diag_damp > 0:
                blk = blk + diag_damp * torch.eye(blk.shape[0], dtype=blk.dtype)
            v["B_blocks"][bi] = blk.cpu()
        ca = max(float(v["count_a"]), 1.0)
        if collect_a_diag:
            v["A_diag"] = (v["A_diag"] / ca).float().cpu()
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
    accum_device: Optional[str] = None,
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
    compute_dev = torch.device(device)
    stats_dev = torch.device(accum_device) if accum_device is not None else compute_dev
    param_grad_state = [(p, p.requires_grad) for p in model.parameters()]
    fn_t0 = time.perf_counter()

    try:
        # Freeze all parameters first to avoid allocating full-model gradients.
        # We re-enable only target low-rank U weights needed to keep backward graph alive.
        for p, _ in param_grad_state:
            p.requires_grad_(False)

        for pair in pairs:
            rank = pair.u_module.weight.shape[1]
            stats[pair.key] = {
                "A_proj": torch.zeros((rank, rank), dtype=torch.float32, device=stats_dev),
                "G_proj": torch.zeros((rank, rank), dtype=torch.float32, device=stats_dev),
                "count_a": 0.0,
                "count_g": 0.0,
            }

        def mk_fwd_pre(k: str, subset_proj_cache: Dict[str, Dict[str, torch.Tensor]]):
            def _hook(module, inputs):
                x = inputs[0].detach().float()
                x_flat = x.reshape(-1, x.shape[-1])
                if "W_inv" in subset_proj_cache[k]:
                    x_flat = x_flat.matmul(subset_proj_cache[k]["W_inv"])
                V_proj = subset_proj_cache[k]["V_proj"]
                x_proj = x_flat if x_flat.dtype == V_proj.dtype else x_flat.to(dtype=V_proj.dtype)
                z = x_proj.matmul(V_proj.transpose(0, 1))
                cov = z.transpose(0, 1).matmul(z).to(dtype=torch.float32)
                if stats_dev.type == "cpu":
                    stats[k]["A_proj"] += cov.cpu()
                else:
                    stats[k]["A_proj"] += cov.to(device=stats_dev)
                stats[k]["count_a"] += float(z.shape[0])

            return _hook

        def mk_bwd(k: str, subset_proj_cache: Dict[str, Dict[str, torch.Tensor]]):
            def _hook(module, grad_input, grad_output):
                g = grad_output[0].detach().float()
                g_flat = g.reshape(-1, g.shape[-1])
                U_proj = subset_proj_cache[k]["U_proj"]
                g_proj = g_flat if g_flat.dtype == U_proj.dtype else g_flat.to(dtype=U_proj.dtype)
                s = g_proj.matmul(U_proj)
                cov = s.transpose(0, 1).matmul(s).to(dtype=torch.float32)
                if stats_dev.type == "cpu":
                    stats[k]["G_proj"] += cov.cpu()
                else:
                    stats[k]["G_proj"] += cov.to(device=stats_dev)
                stats[k]["count_g"] += float(s.shape[0])

            return _hook

        cached_batches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if layerwise:
            cache_t0 = time.perf_counter()
            cached_batches = []
            for idx, batch in enumerate(dataloader):
                if idx >= nsamples:
                    break
                cached_batches.append((batch[0].detach().cpu(), batch[1].detach().cpu()))
            print(
                f"[KFAC sigma_full] cached {len(cached_batches)} batches for layerwise replay "
                f"in {time.perf_counter() - cache_t0:.2f}s"
            )

        pair_subsets = _group_pairs_by_layer(pairs) if layerwise else [pairs]
        print(
            f"[KFAC sigma_full] start: layerwise={layerwise}, subsets={len(pair_subsets)}, "
            f"pairs={len(pairs)}, nsamples={nsamples}, accum_device={stats_dev}"
        )
        subset_bar = tqdm(
            enumerate(pair_subsets, start=1),
            total=len(pair_subsets),
            desc="KFAC sigma_full",
            dynamic_ncols=True,
        )
        for subset_idx, subset in subset_bar:
            subset_t0 = time.perf_counter()
            layer_idx = _extract_layer_index(subset[0].parent_path) if len(subset) > 0 else None
            subset_name = (
                f"layer-{layer_idx}" if layer_idx is not None else f"subset-{subset_idx}"
            )
            subset_proj_cache: Dict[str, Dict[str, torch.Tensor]] = {}
            for pair in subset:
                pair.u_module.weight.requires_grad_(True)
                U_proj = pair.u_module.weight.detach().to(device=compute_dev)
                V_proj = pair.v_module.weight.detach().to(device=compute_dev)
                if explicit_sigma:
                    _, _, _, active, u_scale, v_scale = _compute_sigma_component_norms(
                        u_weight=U_proj,
                        v_weight=V_proj,
                        eps=sigma_eps,
                    )
                    active_mask_u = active.to(device=compute_dev, dtype=U_proj.dtype).unsqueeze(0)
                    active_mask_v = active.to(device=compute_dev, dtype=V_proj.dtype).unsqueeze(1)
                    u_scale = torch.where(active, u_scale, torch.ones_like(u_scale)).to(device=compute_dev, dtype=U_proj.dtype)
                    v_scale = torch.where(active, v_scale, torch.ones_like(v_scale)).to(device=compute_dev, dtype=V_proj.dtype)
                    U_proj = (U_proj / u_scale.unsqueeze(0)) * active_mask_u
                    V_proj = (V_proj / v_scale.unsqueeze(1)) * active_mask_v
                entry: Dict[str, torch.Tensor] = {"U_proj": U_proj, "V_proj": V_proj}
                if whiten_inv is not None and pair.key in whiten_inv:
                    entry["W_inv"] = whiten_inv[pair.key].to(device=compute_dev, dtype=torch.float32)
                subset_proj_cache[pair.key] = entry

            handles = []
            for pair in subset:
                handles.append(pair.v_module.register_forward_pre_hook(mk_fwd_pre(pair.key, subset_proj_cache)))
                handles.append(pair.u_module.register_full_backward_hook(mk_bwd(pair.key, subset_proj_cache)))

            if cached_batches is not None:
                run_batches = cached_batches
            else:
                run_batches = []
                for idx, batch in enumerate(dataloader):
                    if idx >= nsamples:
                        break
                    run_batches.append((batch[0], batch[1]))

            print(
                f"[KFAC sigma_full] {subset_name}: pairs={len(subset)}, batches={len(run_batches)}"
            )
            batch_bar = tqdm(
                enumerate(run_batches, start=1),
                total=len(run_batches),
                desc=f"{subset_name} batches",
                leave=False,
                dynamic_ncols=True,
            )
            for batch_idx, (input_ids, labels) in batch_bar:
                batch_t0 = time.perf_counter()
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                model.zero_grad(set_to_none=True)
                out = model(input_ids=input_ids, labels=labels, use_cache=False)
                loss = out.loss
                loss.backward()
                if batch_idx == 1 or batch_idx == len(run_batches) or batch_idx % max(1, len(run_batches) // 4) == 0:
                    batch_bar.set_postfix(
                        loss=f"{float(loss.detach().item()):.4f}",
                        sec=f"{time.perf_counter() - batch_t0:.2f}",
                    )
                del out, loss

            for h in handles:
                h.remove()
            model.zero_grad(set_to_none=True)
            for pair in subset:
                pair.u_module.weight.requires_grad_(False)
            subset_proj_cache.clear()
            subset_sec = time.perf_counter() - subset_t0
            mean_sec = subset_sec / max(len(run_batches), 1)
            print(
                f"[KFAC sigma_full] done {subset_name}: {subset_sec:.2f}s total, "
                f"{mean_sec:.2f}s/batch"
            )
            subset_bar.set_postfix(layer=subset_name, sec=f"{subset_sec:.1f}")

        for key, v in stats.items():
            ca = max(float(v["count_a"]), 1.0)
            cg = max(float(v["count_g"]), 1.0)
            v["A_proj"] = (v["A_proj"] / ca).float().cpu()
            v["G_proj"] = (v["G_proj"] / cg).float().cpu()
            del v["count_a"]
            del v["count_g"]

        print(f"[KFAC sigma_full] finished in {time.perf_counter() - fn_t0:.2f}s")
        return stats
    finally:
        for p, req in param_grad_state:
            p.requires_grad_(req)
        _restore_kfac_mode(model, use_cache, prev_training, gc_enabled)


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


def solve_budgeted_multilevel_quadratic(
    pairs: List[PairModules],
    fisher_sigma: Dict[str, torch.Tensor],
    low_bit: int = 4,
    high_bit: int = 8,
    avg_bit: float = 4.5,
    sigma_calib: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    sigma_eps: float = 1e-12,
    drop_bit: float = 0.0,
    min_keep_ratio: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Unified drop/low/high allocation with sigma-space quadratic objective.
    State semantics per component:
      drop -> deterministic perturbation delta_sigma = -sigma
      low/high -> zero-mean quantization perturbation with variance proxy
    Objective approximates expected loss:
      E[Delta L] = 1/2 * d^T F d + 1/2 * sum_i F_ii * var_i
    where d is deterministic perturbation vector.
    """
    if high_bit <= low_bit:
        raise ValueError("high_bit must be larger than low_bit")
    if drop_bit < 0:
        raise ValueError("drop_bit must be non-negative")
    if not (0.0 <= min_keep_ratio <= 1.0):
        raise ValueError("min_keep_ratio must be in [0, 1]")

    target_avg = min(max(avg_bit, float(drop_bit)), float(high_bit))
    states: Dict[str, Dict[str, object]] = {}
    alloc: Dict[str, torch.Tensor] = {}
    total_params = 0.0
    used = 0.0
    noise_low = _quant_noise_proxy(low_bit)
    noise_high = _quant_noise_proxy(high_bit)

    for pair in pairs:
        out_dim, rank = pair.u_module.weight.shape
        _, in_dim = pair.v_module.weight.shape
        params_per_component = float(out_dim + in_dim)
        alloc[pair.key] = torch.full((rank,), MP_STATE_DROP, dtype=torch.uint8)

        if pair.key not in fisher_sigma:
            continue
        F = fisher_sigma[pair.key].float().cpu()
        if F.shape != (rank, rank):
            continue
        sigma_vec = _estimate_sigma_proxy(pair).float().cpu()
        active = sigma_vec > sigma_eps
        active_count = int(active.sum().item())
        if active_count == 0:
            continue
        total_params += params_per_component * float(active_count)

        if sigma_calib is not None and pair.key in sigma_calib:
            low_noise = sigma_calib[pair.key]["low"].float().clamp_min(0.0).cpu()
            high_noise = sigma_calib[pair.key]["high"].float().clamp_min(0.0).cpu()
        else:
            low_noise = torch.full((rank,), noise_low, dtype=torch.float32)
            high_noise = torch.full((rank,), noise_high, dtype=torch.float32)

        F = 0.5 * (F + F.transpose(0, 1))
        diag = torch.diag(F)
        d = torch.zeros(rank, dtype=torch.float32)
        d[active] = -sigma_vec[active]
        Fd = F.matmul(d)

        low_var = (sigma_vec * sigma_vec) * low_noise
        high_var = (sigma_vec * sigma_vec) * high_noise
        low_var = torch.where(active, low_var, torch.zeros_like(low_var))
        high_var = torch.where(active, high_var, torch.zeros_like(high_var))
        var = torch.zeros(rank, dtype=torch.float32)
        state = torch.full((rank,), MP_STATE_DROP, dtype=torch.uint8)

        states[pair.key] = {
            "F": F,
            "diag": diag,
            "d": d,
            "Fd": Fd,
            "var": var,
            "low_var": low_var,
            "high_var": high_var,
            "state": state,
            "sigma": sigma_vec,
            "active": active,
            "params_per_component": params_per_component,
            "cost_drop": params_per_component * float(drop_bit),
            "cost_low": params_per_component * float(low_bit),
            "cost_high": params_per_component * float(high_bit),
        }

    extra_budget = total_params * (target_avg - float(drop_bit))

    def _apply_action(st: Dict[str, object], idx: int, to_state: int) -> float:
        cur = int(st["state"][idx].item())  # type: ignore[index]
        if cur == to_state:
            return 0.0
        delta_d = 0.0
        delta_var = 0.0
        sigma_i = float(st["sigma"][idx].item())  # type: ignore[index]
        low_var_i = float(st["low_var"][idx].item())  # type: ignore[index]
        high_var_i = float(st["high_var"][idx].item())  # type: ignore[index]
        diag_i = float(st["diag"][idx].item())  # type: ignore[index]
        Fd_i = float(st["Fd"][idx].item())  # type: ignore[index]
        cost_drop = float(st["cost_drop"])
        cost_low = float(st["cost_low"])
        cost_high = float(st["cost_high"])

        if cur == MP_STATE_DROP and to_state == MP_STATE_LOW:
            delta_d = sigma_i
            delta_var = low_var_i
            delta_cost = cost_low - cost_drop
        elif cur == MP_STATE_DROP and to_state == MP_STATE_HIGH:
            delta_d = sigma_i
            delta_var = high_var_i
            delta_cost = cost_high - cost_drop
        elif cur == MP_STATE_LOW and to_state == MP_STATE_HIGH:
            delta_var = high_var_i - low_var_i
            delta_cost = cost_high - cost_low
        else:
            return 0.0

        if abs(delta_d) > 0:
            st["d"][idx] += delta_d  # type: ignore[index]
            st["Fd"] += st["F"][:, idx] * delta_d  # type: ignore[index]
        if abs(delta_var) > 0:
            st["var"][idx] += delta_var  # type: ignore[index]
        st["state"][idx] = to_state  # type: ignore[index]
        return float(delta_cost)

    def _best_action_for_state(
        st: Dict[str, object], remain_budget: float
    ) -> Optional[Tuple[int, int, float, float, float]]:
        """
        Return best feasible action as:
          (idx, to_state, gain, density, delta_cost)
        Vectorized over all components in this pair.
        """
        if remain_budget <= 0.0:
            return None
        eps_budget = 1e-9
        active = st["active"]  # type: ignore[assignment]
        state = st["state"]  # type: ignore[assignment]
        sigma = st["sigma"]  # type: ignore[assignment]
        low_var = st["low_var"]  # type: ignore[assignment]
        high_var = st["high_var"]  # type: ignore[assignment]
        diag = st["diag"]  # type: ignore[assignment]
        Fd = st["Fd"]  # type: ignore[assignment]
        cost_drop = float(st["cost_drop"])
        cost_low = float(st["cost_low"])
        cost_high = float(st["cost_high"])

        inf_neg = torch.tensor(-float("inf"), dtype=torch.float32)
        best_idx = -1
        best_to_state = MP_STATE_DROP
        best_gain = 0.0
        best_density = 0.0
        best_cost = 0.0

        # Drop -> Low
        cost_dl = cost_low - cost_drop
        if cost_dl > 0.0 and cost_dl <= remain_budget + eps_budget:
            mask_dl = active & (state == MP_STATE_DROP)
            if torch.any(mask_dl):
                gain_dl = -(sigma * Fd + 0.5 * (sigma * sigma) * diag + 0.5 * low_var)
                gain_dl = torch.where(mask_dl, gain_dl, inf_neg)
                max_gain_dl, idx_dl = torch.max(gain_dl, dim=0)
                gain_val = float(max_gain_dl.item())
                if gain_val > 0.0:
                    density = gain_val / cost_dl
                    best_idx = int(idx_dl.item())
                    best_to_state = MP_STATE_LOW
                    best_gain = gain_val
                    best_density = density
                    best_cost = cost_dl

        # Drop -> High
        cost_dh = cost_high - cost_drop
        if cost_dh > 0.0 and cost_dh <= remain_budget + eps_budget:
            mask_dh = active & (state == MP_STATE_DROP)
            if torch.any(mask_dh):
                gain_dh = -(sigma * Fd + 0.5 * (sigma * sigma) * diag + 0.5 * high_var)
                gain_dh = torch.where(mask_dh, gain_dh, inf_neg)
                max_gain_dh, idx_dh = torch.max(gain_dh, dim=0)
                gain_val = float(max_gain_dh.item())
                if gain_val > 0.0:
                    density = gain_val / cost_dh
                    if density > best_density:
                        best_idx = int(idx_dh.item())
                        best_to_state = MP_STATE_HIGH
                        best_gain = gain_val
                        best_density = density
                        best_cost = cost_dh

        # Low -> High
        cost_lh = cost_high - cost_low
        if cost_lh > 0.0 and cost_lh <= remain_budget + eps_budget:
            mask_lh = active & (state == MP_STATE_LOW)
            if torch.any(mask_lh):
                gain_lh = -(0.5 * (high_var - low_var) * diag)
                gain_lh = torch.where(mask_lh, gain_lh, inf_neg)
                max_gain_lh, idx_lh = torch.max(gain_lh, dim=0)
                gain_val = float(max_gain_lh.item())
                if gain_val > 0.0:
                    density = gain_val / cost_lh
                    if density > best_density:
                        best_idx = int(idx_lh.item())
                        best_to_state = MP_STATE_HIGH
                        best_gain = gain_val
                        best_density = density
                        best_cost = cost_lh

        if best_idx < 0 or best_gain <= 0.0 or best_density <= 0.0:
            return None
        return best_idx, best_to_state, best_gain, best_density, best_cost

    if min_keep_ratio > 0.0:
        for key, st in states.items():
            active = st["active"]  # type: ignore[assignment]
            active_idx = torch.where(active)[0]
            if active_idx.numel() == 0:
                continue
            keep_count = int(math.ceil(float(active_idx.numel()) * min_keep_ratio))
            if keep_count <= 0:
                continue
            sigma_vec = st["sigma"]  # type: ignore[assignment]
            diag = st["diag"]  # type: ignore[assignment]
            keep_score = (sigma_vec * sigma_vec) * diag
            keep_score = keep_score.clone()
            keep_score[~active] = -float("inf")
            topk = min(keep_count, int(active_idx.numel()))
            chosen = torch.topk(keep_score, k=topk, largest=True).indices
            for idx_t in chosen:
                idx = int(idx_t.item())
                if int(st["state"][idx].item()) != MP_STATE_DROP:  # type: ignore[index]
                    continue
                used += _apply_action(st, idx, MP_STATE_LOW)

    if used > extra_budget + 1e-9:
        print(
            "Warning: min_keep_ratio bootstrap exceeds target budget; "
            f"used_bits={used:.2f}, budget_bits={extra_budget:.2f}"
        )

    layer_best: Dict[str, Optional[Tuple[int, int, float, float, float]]] = {}
    dirty_keys = set(states.keys())

    while True:
        remain_budget = extra_budget - used
        if remain_budget <= 1e-9:
            break

        # Lazy update: only recompute modified (or budget-invalidated) layers.
        if dirty_keys:
            for key in list(dirty_keys):
                layer_best[key] = _best_action_for_state(states[key], remain_budget)
            dirty_keys.clear()

        best_key = None
        best_idx = -1
        best_to_state = MP_STATE_DROP
        best_gain = 0.0
        best_density = 0.0
        best_cost = 0.0
        budget_invalid = []

        for key, cand in layer_best.items():
            if cand is None:
                continue
            idx, to_state, gain, density, delta_cost = cand
            if delta_cost > remain_budget + 1e-9:
                budget_invalid.append(key)
                continue
            if density > best_density:
                best_key = key
                best_idx = idx
                best_to_state = to_state
                best_gain = gain
                best_density = density
                best_cost = delta_cost

        if best_key is None or best_idx < 0 or best_gain <= 0.0:
            if budget_invalid:
                dirty_keys.update(budget_invalid)
                continue
            break

        st = states[best_key]
        used += _apply_action(st, best_idx, best_to_state)
        if best_cost <= 0.0:
            break
        dirty_keys.add(best_key)

    for key, st in states.items():
        alloc[key] = st["state"].clone()  # type: ignore[assignment]
    return alloc


def _quantize_vec_symmetric(vec: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = float((2 ** (bits - 1)) - 1)
    if qmax <= 0:
        return torch.zeros_like(vec)
    scale = vec.abs().max().clamp_min(1e-8) / qmax
    q = torch.round(vec / scale).clamp(-qmax, qmax)
    return q * scale


def _can_use_int8_gemm(device: torch.device, use_int8_kernel: bool) -> bool:
    return bool(use_int8_kernel and device.type == "cuda" and hasattr(torch, "_int_mm"))


def _can_use_int4_gemm(device: torch.device, use_int4_kernel: bool) -> bool:
    return bool(use_int4_kernel and device.type == "cuda" and bnb is not None)


def _make_bnb_linear4(
    weight: torch.Tensor,
    device: torch.device,
    compute_dtype: torch.dtype = torch.float16,
    quant_type: str = "nf4",
) -> Optional[nn.Module]:
    if bnb is None:
        return None
    linear4 = bnb.nn.Linear4bit(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        compute_dtype=compute_dtype,
        compress_statistics=True,
        quant_type=quant_type,
    )
    # Build Params4bit on CPU first, then move module to CUDA to initialize quant_state.
    w = weight.to(device="cpu", dtype=compute_dtype).contiguous()
    linear4.weight = bnb.nn.Params4bit(
        w,
        requires_grad=False,
        compress_statistics=True,
        quant_type=quant_type,
    )
    linear4 = linear4.to(device)
    linear4.eval()
    return linear4


def _run_bnb_linear4(linear4: Optional[nn.Module], x: torch.Tensor) -> Optional[torch.Tensor]:
    if linear4 is None:
        return None
    try:
        return linear4(x)
    except AssertionError:
        # bitsandbytes may miss quant_state when module/device state changes unexpectedly.
        # Retry after refreshing device placement; return None on persistent failure.
        try:
            linear4.to(x.device)
            return linear4(x)
        except Exception:
            return None


def _quantize_activation_int8_2d(x2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Chunked quantization to limit temporary memory spikes on long sequences / large batches.
    rows = x2d.shape[0]
    if rows == 0:
        return torch.empty_like(x2d, dtype=torch.int8), torch.empty((0,), device=x2d.device, dtype=torch.float32)
    chunk_rows = 4096
    xq = torch.empty_like(x2d, dtype=torch.int8)
    scales = torch.empty((rows,), device=x2d.device, dtype=torch.float32)
    for s in range(0, rows, chunk_rows):
        e = min(s + chunk_rows, rows)
        xb = x2d[s:e]
        xmax = torch.amax(xb, dim=1)
        xmin = torch.amin(xb, dim=1)
        absmax = torch.maximum(xmax, -xmin).to(torch.float32).clamp_min_(1e-8)
        sc = absmax / 127.0
        q = torch.round(xb.to(torch.float32) / sc.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        xq[s:e] = q
        scales[s:e] = sc
    return xq.contiguous(), scales


def _int8_linear_dynamic_act(
    x2d: torch.Tensor,
    weight_q_t: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    # Fallback path for shapes unsupported by torch._int_mm.
    # Keep numerically equivalent semantics: y = x @ (W_q * s).
    if weight_q_t.ndim != 2:
        return torch.zeros((x2d.shape[0], 0), device=x2d.device, dtype=torch.float32)
    if weight_q_t.shape[0] == 0 or weight_q_t.shape[1] == 0:
        return torch.zeros((x2d.shape[0], int(weight_q_t.shape[1])), device=x2d.device, dtype=torch.float32)
    if weight_q_t.shape[1] % 8 != 0:
        w = weight_q_t.to(dtype=torch.float32) * weight_scale.unsqueeze(0).to(dtype=torch.float32, device=weight_q_t.device)
        return x2d.to(dtype=torch.float32).matmul(w)

    xq, x_scale = _quantize_activation_int8_2d(x2d)
    try:
        y_i32 = torch._int_mm(xq, weight_q_t)
        y = y_i32.to(torch.float32)
        y.mul_(x_scale.unsqueeze(1))
        y.mul_(weight_scale.unsqueeze(0))
        return y
    except RuntimeError:
        # Some GPUs / torch builds enforce extra shape constraints in _int_mm.
        # Safe fallback to dequantized matmul for this invocation.
        w = weight_q_t.to(dtype=torch.float32) * weight_scale.unsqueeze(0).to(dtype=torch.float32, device=weight_q_t.device)
        return x2d.to(dtype=torch.float32).matmul(w)


def _pad_int8_chain_rank(
    v_q_t: torch.Tensor,
    v_s: torch.Tensor,
    u_q_t: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    multiple: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Pad rank dimension so torch._int_mm shape constraints are satisfied.
    v_q_t: [in_features, rank]
    v_s:   [rank]
    u_q_t: [rank, out_features]
    sigma: [rank] (optional, for explicit-sigma path)
    """
    if multiple <= 1:
        return v_q_t, v_s, u_q_t, sigma
    rank = int(v_q_t.shape[1])
    if rank == 0 or rank % multiple == 0:
        return v_q_t, v_s, u_q_t, sigma

    pad = multiple - (rank % multiple)
    v_pad = torch.zeros((v_q_t.shape[0], pad), device=v_q_t.device, dtype=v_q_t.dtype)
    u_pad = torch.zeros((pad, u_q_t.shape[1]), device=u_q_t.device, dtype=u_q_t.dtype)
    # Any positive finite scale is fine for zero quantized weights.
    s_pad = torch.ones((pad,), device=v_s.device, dtype=v_s.dtype)

    v_q_t = torch.cat([v_q_t, v_pad], dim=1)
    v_s = torch.cat([v_s, s_pad], dim=0)
    u_q_t = torch.cat([u_q_t, u_pad], dim=0)

    if sigma is not None:
        sigma_pad = torch.zeros((pad,), device=sigma.device, dtype=sigma.dtype)
        sigma = torch.cat([sigma, sigma_pad], dim=0)

    return v_q_t, v_s, u_q_t, sigma


def _pad_dense_chain_rank(
    v: torch.Tensor,
    u: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    multiple: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Pad dense low-rank chain (V, U, [sigma]) along rank dimension.
    v: [rank, in_features], u: [out_features, rank], sigma: [rank] (optional)
    """
    if multiple <= 1:
        return v, u, sigma
    rank = int(v.shape[0])
    if rank == 0 or rank % multiple == 0:
        return v, u, sigma

    pad = multiple - (rank % multiple)
    v_pad = torch.zeros((pad, v.shape[1]), device=v.device, dtype=v.dtype)
    u_pad = torch.zeros((u.shape[0], pad), device=u.device, dtype=u.dtype)
    v = torch.cat([v, v_pad], dim=0)
    u = torch.cat([u, u_pad], dim=1)
    if sigma is not None:
        sigma_pad = torch.zeros((pad,), device=sigma.device, dtype=sigma.dtype)
        sigma = torch.cat([sigma, sigma_pad], dim=0)
    return v, u, sigma


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
        use_int8_kernel: bool = True,
        use_int4_kernel: bool = False,
        int4_quant_type: str = "nf4",
    ):
        super().__init__()
        meta_dtype = torch.float16
        self.high_bit = high_bit
        self.low_bit = low_bit
        self.use_int8_kernel = use_int8_kernel
        self.use_int4_kernel = use_int4_kernel
        self.int4_quant_type = int4_quant_type
        self.runtime_cache_persistent = False
        self.out_features = u_weight.shape[0]
        self.in_features = v_weight.shape[1]
        self._runtime_u: Optional[torch.Tensor] = None
        self._runtime_v: Optional[torch.Tensor] = None
        self._runtime_uh: Optional[torch.Tensor] = None
        self._runtime_vh: Optional[torch.Tensor] = None
        self._runtime_ul: Optional[torch.Tensor] = None
        self._runtime_vl: Optional[torch.Tensor] = None
        self._runtime_uhq_t: Optional[torch.Tensor] = None
        self._runtime_vhq_t: Optional[torch.Tensor] = None
        self._runtime_uhs: Optional[torch.Tensor] = None
        self._runtime_vhs: Optional[torch.Tensor] = None
        self._runtime_ulq_t: Optional[torch.Tensor] = None
        self._runtime_vlq_t: Optional[torch.Tensor] = None
        self._runtime_uls: Optional[torch.Tensor] = None
        self._runtime_vls: Optional[torch.Tensor] = None
        self._runtime_vh4: Optional[nn.Module] = None
        self._runtime_uh4: Optional[nn.Module] = None
        self._runtime_vl4: Optional[nn.Module] = None
        self._runtime_ul4: Optional[nn.Module] = None
        self._runtime_device: Optional[torch.device] = None
        self._runtime_dtype: Optional[torch.dtype] = None
        self.register_buffer("high_idx", high_idx.to(torch.long), persistent=False)
        self.register_buffer("low_idx", low_idx.to(torch.long), persistent=False)

        if self.high_idx.numel() > 0:
            uh, vh = u_weight[:, self.high_idx], v_weight[self.high_idx, :]
            uh_q, uh_s = self._q_per_row(uh, high_bit)
            vh_q, vh_s = self._q_per_row(vh, high_bit)
            self.register_buffer("uh_q", uh_q, persistent=True)
            self.register_buffer("uh_s", uh_s.to(dtype=meta_dtype), persistent=True)
            self.register_buffer("vh_q", vh_q, persistent=True)
            self.register_buffer("vh_s", vh_s.to(dtype=meta_dtype), persistent=True)
        else:
            self.register_buffer("uh_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("uh_s", torch.empty(0, dtype=meta_dtype), persistent=True)
            self.register_buffer("vh_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("vh_s", torch.empty(0, dtype=meta_dtype), persistent=True)

        if self.low_idx.numel() > 0:
            ul, vl = u_weight[:, self.low_idx], v_weight[self.low_idx, :]
            ul_q, ul_s = self._q_per_row(ul, low_bit)
            vl_q, vl_s = self._q_per_row(vl, low_bit)
            self.register_buffer("ul_q", ul_q, persistent=True)
            self.register_buffer("ul_s", ul_s.to(dtype=meta_dtype), persistent=True)
            self.register_buffer("vl_q", vl_q, persistent=True)
            self.register_buffer("vl_s", vl_s.to(dtype=meta_dtype), persistent=True)
        else:
            self.register_buffer("ul_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("ul_s", torch.empty(0, dtype=meta_dtype), persistent=True)
            self.register_buffer("vl_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("vl_s", torch.empty(0, dtype=meta_dtype), persistent=True)

        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone().to(dtype=meta_dtype), persistent=True)

    @staticmethod
    def _q_per_row(w: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        qmax = float((2 ** (bits - 1)) - 1)
        w = w.float()
        s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(w / s).clamp(-qmax, qmax).to(torch.int8)
        return q.cpu(), s.squeeze(1).to(dtype=torch.float16).cpu()

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

        self._runtime_uhq_t = None
        self._runtime_vhq_t = None
        self._runtime_uhs = None
        self._runtime_vhs = None
        self._runtime_uh = None
        self._runtime_vh = None
        self._runtime_ulq_t = None
        self._runtime_vlq_t = None
        self._runtime_uls = None
        self._runtime_vls = None
        self._runtime_ul = None
        self._runtime_vl = None
        self._runtime_vh4 = None
        self._runtime_uh4 = None
        self._runtime_vl4 = None
        self._runtime_ul4 = None

        parts_u = []
        parts_v = []
        if self.uh_q.numel() > 0:
            uh = self._deq_per_row(self.uh_q, self.uh_s, device, dtype)
            vh = self._deq_per_row(self.vh_q, self.vh_s, device, dtype)
            self._runtime_uh = uh
            self._runtime_vh = vh
            parts_u.append(uh)
            parts_v.append(vh)
            u_q = self.uh_q.to(device=device)
            v_q = self.vh_q.to(device=device)
            self._runtime_uhq_t = u_q.transpose(0, 1).contiguous()
            self._runtime_vhq_t = v_q.transpose(0, 1).contiguous()
            self._runtime_uhs = self.uh_s.to(device=device, dtype=torch.float32)
            self._runtime_vhs = self.vh_s.to(device=device, dtype=torch.float32)
            self._runtime_vhq_t, self._runtime_vhs, self._runtime_uhq_t, _ = _pad_int8_chain_rank(
                self._runtime_vhq_t,
                self._runtime_vhs,
                self._runtime_uhq_t,
                sigma=None,
                multiple=8,
            )
            if _can_use_int4_gemm(device, self.use_int4_kernel) and self.high_bit == 4:
                vh_i4, uh_i4, _ = _pad_dense_chain_rank(vh.float(), uh.float(), sigma=None, multiple=8)
                self._runtime_vh4 = _make_bnb_linear4(
                    vh_i4,
                    device=device,
                    compute_dtype=torch.float16,
                    quant_type=self.int4_quant_type,
                )
                self._runtime_uh4 = _make_bnb_linear4(
                    uh_i4,
                    device=device,
                    compute_dtype=torch.float16,
                    quant_type=self.int4_quant_type,
                )
        if self.ul_q.numel() > 0:
            ul = self._deq_per_row(self.ul_q, self.ul_s, device, dtype)
            vl = self._deq_per_row(self.vl_q, self.vl_s, device, dtype)
            self._runtime_ul = ul
            self._runtime_vl = vl
            parts_u.append(ul)
            parts_v.append(vl)
            u_q = self.ul_q.to(device=device)
            v_q = self.vl_q.to(device=device)
            self._runtime_ulq_t = u_q.transpose(0, 1).contiguous()
            self._runtime_vlq_t = v_q.transpose(0, 1).contiguous()
            self._runtime_uls = self.ul_s.to(device=device, dtype=torch.float32)
            self._runtime_vls = self.vl_s.to(device=device, dtype=torch.float32)
            self._runtime_vlq_t, self._runtime_vls, self._runtime_ulq_t, _ = _pad_int8_chain_rank(
                self._runtime_vlq_t,
                self._runtime_vls,
                self._runtime_ulq_t,
                sigma=None,
                multiple=8,
            )
            if _can_use_int4_gemm(device, self.use_int4_kernel) and self.low_bit == 4:
                vl_i4, ul_i4, _ = _pad_dense_chain_rank(vl.float(), ul.float(), sigma=None, multiple=8)
                self._runtime_vl4 = _make_bnb_linear4(
                    vl_i4,
                    device=device,
                    compute_dtype=torch.float16,
                    quant_type=self.int4_quant_type,
                )
                self._runtime_ul4 = _make_bnb_linear4(
                    ul_i4,
                    device=device,
                    compute_dtype=torch.float16,
                    quant_type=self.int4_quant_type,
                )
        if len(parts_u) == 0:
            self._runtime_u = torch.zeros((self.out_features, 0), device=device, dtype=dtype)
            self._runtime_v = torch.zeros((0, self.in_features), device=device, dtype=dtype)
        elif len(parts_u) == 1:
            self._runtime_u = parts_u[0]
            self._runtime_v = parts_v[0]
        else:
            self._runtime_u = torch.cat(parts_u, dim=1)
            self._runtime_v = torch.cat(parts_v, dim=0)
        self._runtime_device = device
        self._runtime_dtype = dtype

    def clear_runtime_cache(self):
        self._runtime_u = None
        self._runtime_v = None
        self._runtime_uh = None
        self._runtime_vh = None
        self._runtime_ul = None
        self._runtime_vl = None
        self._runtime_uhq_t = None
        self._runtime_vhq_t = None
        self._runtime_uhs = None
        self._runtime_vhs = None
        self._runtime_ulq_t = None
        self._runtime_vlq_t = None
        self._runtime_uls = None
        self._runtime_vls = None
        self._runtime_vh4 = None
        self._runtime_uh4 = None
        self._runtime_vl4 = None
        self._runtime_ul4 = None
        self._runtime_device = None
        self._runtime_dtype = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        dev = x.device
        self._build_runtime_cache(dev, dtype)
        if self._runtime_u is None or self._runtime_v is None or self._runtime_v.numel() == 0:
            out = torch.zeros((*x.shape[:-1], self.out_features), device=dev, dtype=dtype)
        elif _can_use_int8_gemm(dev, self.use_int8_kernel) or _can_use_int4_gemm(dev, self.use_int4_kernel):
            x2d = x.reshape(-1, x.shape[-1]).contiguous()
            out2d = torch.zeros((x2d.shape[0], self.out_features), device=dev, dtype=torch.float32)
            x4 = x2d.to(dtype=torch.float16)
            if self._runtime_vh is not None and self._runtime_uh is not None:
                if self._runtime_vh4 is not None and self._runtime_uh4 is not None:
                    zh = _run_bnb_linear4(self._runtime_vh4, x4)
                    oh = _run_bnb_linear4(self._runtime_uh4, zh) if zh is not None else None
                    if oh is not None:
                        out2d.add_(oh.to(dtype=torch.float32))
                    else:
                        self._runtime_vh4 = None
                        self._runtime_uh4 = None
                elif (
                    _can_use_int8_gemm(dev, self.use_int8_kernel)
                    and self._runtime_uhq_t is not None
                    and self._runtime_vhq_t is not None
                    and self._runtime_uhs is not None
                    and self._runtime_vhs is not None
                ):
                    zh = _int8_linear_dynamic_act(x2d, self._runtime_vhq_t, self._runtime_vhs)
                    out2d.add_(_int8_linear_dynamic_act(zh, self._runtime_uhq_t, self._runtime_uhs))
                else:
                    zh = F.linear(x2d.to(dtype=self._runtime_vh.dtype), self._runtime_vh)
                    oh = F.linear(zh, self._runtime_uh)
                    out2d.add_(oh.to(dtype=torch.float32))
            if self._runtime_vl is not None and self._runtime_ul is not None:
                if self._runtime_vl4 is not None and self._runtime_ul4 is not None:
                    zl = _run_bnb_linear4(self._runtime_vl4, x4)
                    ol = _run_bnb_linear4(self._runtime_ul4, zl) if zl is not None else None
                    if ol is not None:
                        out2d.add_(ol.to(dtype=torch.float32))
                    else:
                        self._runtime_vl4 = None
                        self._runtime_ul4 = None
                elif (
                    _can_use_int8_gemm(dev, self.use_int8_kernel)
                    and self._runtime_ulq_t is not None
                    and self._runtime_vlq_t is not None
                    and self._runtime_uls is not None
                    and self._runtime_vls is not None
                ):
                    zl = _int8_linear_dynamic_act(x2d, self._runtime_vlq_t, self._runtime_vls)
                    out2d.add_(_int8_linear_dynamic_act(zl, self._runtime_ulq_t, self._runtime_uls))
                else:
                    zl = F.linear(x2d.to(dtype=self._runtime_vl.dtype), self._runtime_vl)
                    ol = F.linear(zl, self._runtime_ul)
                    out2d.add_(ol.to(dtype=torch.float32))
            out = out2d.reshape(*x.shape[:-1], self.out_features).to(dtype=dtype)
        else:
            z = F.linear(x, self._runtime_v)
            out = F.linear(z, self._runtime_u)
        if self.bias is not None:
            out = out + self.bias.to(device=dev, dtype=dtype)
        if not self.runtime_cache_persistent:
            self.clear_runtime_cache()
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
        use_int8_kernel: bool = True,
        use_int4_kernel: bool = False,
        int4_quant_type: str = "nf4",
    ):
        super().__init__()
        meta_dtype = torch.float16
        self.high_bit = high_bit
        self.low_bit = low_bit
        self.use_int8_kernel = use_int8_kernel
        self.use_int4_kernel = use_int4_kernel
        self.int4_quant_type = int4_quant_type
        self.runtime_cache_persistent = False
        self.out_features = u_basis.shape[0]
        self.in_features = v_basis.shape[1]
        self._runtime_u: Optional[torch.Tensor] = None
        self._runtime_v: Optional[torch.Tensor] = None
        self._runtime_uh: Optional[torch.Tensor] = None
        self._runtime_vh: Optional[torch.Tensor] = None
        self._runtime_ul: Optional[torch.Tensor] = None
        self._runtime_vl: Optional[torch.Tensor] = None
        self._runtime_uhq_t: Optional[torch.Tensor] = None
        self._runtime_vhq_t: Optional[torch.Tensor] = None
        self._runtime_uhs: Optional[torch.Tensor] = None
        self._runtime_vhs: Optional[torch.Tensor] = None
        self._runtime_ulq_t: Optional[torch.Tensor] = None
        self._runtime_vlq_t: Optional[torch.Tensor] = None
        self._runtime_uls: Optional[torch.Tensor] = None
        self._runtime_vls: Optional[torch.Tensor] = None
        self._runtime_sh: Optional[torch.Tensor] = None
        self._runtime_sl: Optional[torch.Tensor] = None
        self._runtime_vh4: Optional[nn.Module] = None
        self._runtime_uh4: Optional[nn.Module] = None
        self._runtime_vl4: Optional[nn.Module] = None
        self._runtime_ul4: Optional[nn.Module] = None
        self._runtime_sigma: Optional[torch.Tensor] = None
        self._runtime_device: Optional[torch.device] = None
        self._runtime_dtype: Optional[torch.dtype] = None
        self.register_buffer("high_idx", high_idx.to(torch.long), persistent=False)
        self.register_buffer("low_idx", low_idx.to(torch.long), persistent=False)

        if self.high_idx.numel() > 0:
            uh = u_basis[:, self.high_idx]
            vh = v_basis[self.high_idx, :]
            sh = sigma[self.high_idx].to(dtype=meta_dtype).cpu()
            uh_q, uh_s = self._q_per_row(uh, high_bit)
            vh_q, vh_s = self._q_per_row(vh, high_bit)
            self.register_buffer("uh_q", uh_q, persistent=True)
            self.register_buffer("uh_s", uh_s.to(dtype=meta_dtype), persistent=True)
            self.register_buffer("vh_q", vh_q, persistent=True)
            self.register_buffer("vh_s", vh_s.to(dtype=meta_dtype), persistent=True)
            self.register_buffer("sh", sh, persistent=True)
        else:
            self.register_buffer("uh_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("uh_s", torch.empty(0, dtype=meta_dtype), persistent=True)
            self.register_buffer("vh_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("vh_s", torch.empty(0, dtype=meta_dtype), persistent=True)
            self.register_buffer("sh", torch.empty(0, dtype=meta_dtype), persistent=True)

        if self.low_idx.numel() > 0:
            ul = u_basis[:, self.low_idx]
            vl = v_basis[self.low_idx, :]
            sl = sigma[self.low_idx].to(dtype=meta_dtype).cpu()
            ul_q, ul_s = self._q_per_row(ul, low_bit)
            vl_q, vl_s = self._q_per_row(vl, low_bit)
            self.register_buffer("ul_q", ul_q, persistent=True)
            self.register_buffer("ul_s", ul_s.to(dtype=meta_dtype), persistent=True)
            self.register_buffer("vl_q", vl_q, persistent=True)
            self.register_buffer("vl_s", vl_s.to(dtype=meta_dtype), persistent=True)
            self.register_buffer("sl", sl, persistent=True)
        else:
            self.register_buffer("ul_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("ul_s", torch.empty(0, dtype=meta_dtype), persistent=True)
            self.register_buffer("vl_q", torch.empty(0, dtype=torch.int8), persistent=True)
            self.register_buffer("vl_s", torch.empty(0, dtype=meta_dtype), persistent=True)
            self.register_buffer("sl", torch.empty(0, dtype=meta_dtype), persistent=True)

        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone().to(dtype=meta_dtype), persistent=True)

    @staticmethod
    def _q_per_row(w: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        qmax = float((2 ** (bits - 1)) - 1)
        w = w.float()
        s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(w / s).clamp(-qmax, qmax).to(torch.int8)
        return q.cpu(), s.squeeze(1).to(dtype=torch.float16).cpu()

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

        self._runtime_uhq_t = None
        self._runtime_vhq_t = None
        self._runtime_uhs = None
        self._runtime_vhs = None
        self._runtime_uh = None
        self._runtime_vh = None
        self._runtime_ulq_t = None
        self._runtime_vlq_t = None
        self._runtime_uls = None
        self._runtime_vls = None
        self._runtime_ul = None
        self._runtime_vl = None
        self._runtime_sh = None
        self._runtime_sl = None
        self._runtime_vh4 = None
        self._runtime_uh4 = None
        self._runtime_vl4 = None
        self._runtime_ul4 = None

        parts_u = []
        parts_v = []
        parts_sigma = []
        if self.uh_q.numel() > 0:
            uh = self._deq_per_row(self.uh_q, self.uh_s, device, dtype)
            vh = self._deq_per_row(self.vh_q, self.vh_s, device, dtype)
            self._runtime_uh = uh
            self._runtime_vh = vh
            parts_u.append(uh)
            parts_v.append(vh)
            parts_sigma.append(self.sh.to(device=device, dtype=dtype))
            u_q = self.uh_q.to(device=device)
            v_q = self.vh_q.to(device=device)
            self._runtime_uhq_t = u_q.transpose(0, 1).contiguous()
            self._runtime_vhq_t = v_q.transpose(0, 1).contiguous()
            self._runtime_uhs = self.uh_s.to(device=device, dtype=torch.float32)
            self._runtime_vhs = self.vh_s.to(device=device, dtype=torch.float32)
            self._runtime_sh = self.sh.to(device=device, dtype=torch.float32)
            self._runtime_vhq_t, self._runtime_vhs, self._runtime_uhq_t, self._runtime_sh = _pad_int8_chain_rank(
                self._runtime_vhq_t,
                self._runtime_vhs,
                self._runtime_uhq_t,
                sigma=self._runtime_sh,
                multiple=8,
            )
            if _can_use_int4_gemm(device, self.use_int4_kernel) and self.high_bit == 4:
                vh_i4, uh_i4, sh_i4 = _pad_dense_chain_rank(
                    vh.float(),
                    uh.float(),
                    sigma=self.sh.to(device=device, dtype=torch.float32),
                    multiple=8,
                )
                self._runtime_vh4 = _make_bnb_linear4(
                    vh_i4,
                    device=device,
                    compute_dtype=torch.float16,
                    quant_type=self.int4_quant_type,
                )
                self._runtime_uh4 = _make_bnb_linear4(
                    uh_i4,
                    device=device,
                    compute_dtype=torch.float16,
                    quant_type=self.int4_quant_type,
                )
                if sh_i4 is not None:
                    self._runtime_sh = sh_i4
        if self.ul_q.numel() > 0:
            ul = self._deq_per_row(self.ul_q, self.ul_s, device, dtype)
            vl = self._deq_per_row(self.vl_q, self.vl_s, device, dtype)
            self._runtime_ul = ul
            self._runtime_vl = vl
            parts_u.append(ul)
            parts_v.append(vl)
            parts_sigma.append(self.sl.to(device=device, dtype=dtype))
            u_q = self.ul_q.to(device=device)
            v_q = self.vl_q.to(device=device)
            self._runtime_ulq_t = u_q.transpose(0, 1).contiguous()
            self._runtime_vlq_t = v_q.transpose(0, 1).contiguous()
            self._runtime_uls = self.ul_s.to(device=device, dtype=torch.float32)
            self._runtime_vls = self.vl_s.to(device=device, dtype=torch.float32)
            self._runtime_sl = self.sl.to(device=device, dtype=torch.float32)
            self._runtime_vlq_t, self._runtime_vls, self._runtime_ulq_t, self._runtime_sl = _pad_int8_chain_rank(
                self._runtime_vlq_t,
                self._runtime_vls,
                self._runtime_ulq_t,
                sigma=self._runtime_sl,
                multiple=8,
            )
            if _can_use_int4_gemm(device, self.use_int4_kernel) and self.low_bit == 4:
                vl_i4, ul_i4, sl_i4 = _pad_dense_chain_rank(
                    vl.float(),
                    ul.float(),
                    sigma=self.sl.to(device=device, dtype=torch.float32),
                    multiple=8,
                )
                self._runtime_vl4 = _make_bnb_linear4(
                    vl_i4,
                    device=device,
                    compute_dtype=torch.float16,
                    quant_type=self.int4_quant_type,
                )
                self._runtime_ul4 = _make_bnb_linear4(
                    ul_i4,
                    device=device,
                    compute_dtype=torch.float16,
                    quant_type=self.int4_quant_type,
                )
                if sl_i4 is not None:
                    self._runtime_sl = sl_i4
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

    def clear_runtime_cache(self):
        self._runtime_u = None
        self._runtime_v = None
        self._runtime_uh = None
        self._runtime_vh = None
        self._runtime_ul = None
        self._runtime_vl = None
        self._runtime_uhq_t = None
        self._runtime_vhq_t = None
        self._runtime_uhs = None
        self._runtime_vhs = None
        self._runtime_ulq_t = None
        self._runtime_vlq_t = None
        self._runtime_uls = None
        self._runtime_vls = None
        self._runtime_sh = None
        self._runtime_sl = None
        self._runtime_vh4 = None
        self._runtime_uh4 = None
        self._runtime_vl4 = None
        self._runtime_ul4 = None
        self._runtime_sigma = None
        self._runtime_device = None
        self._runtime_dtype = None

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
        elif _can_use_int8_gemm(dev, self.use_int8_kernel) or _can_use_int4_gemm(dev, self.use_int4_kernel):
            x2d = x.reshape(-1, x.shape[-1]).contiguous()
            out2d = torch.zeros((x2d.shape[0], self.out_features), device=dev, dtype=torch.float32)
            x4 = x2d.to(dtype=torch.float16)
            if self._runtime_vh is not None and self._runtime_uh is not None:
                if self._runtime_vh4 is not None and self._runtime_uh4 is not None:
                    zh = _run_bnb_linear4(self._runtime_vh4, x4)
                    if zh is not None:
                        if self._runtime_sh is not None:
                            zh = zh * self._runtime_sh.unsqueeze(0).to(device=zh.device, dtype=zh.dtype)
                        oh = _run_bnb_linear4(self._runtime_uh4, zh)
                    else:
                        oh = None
                    if oh is not None:
                        out2d.add_(oh.to(dtype=torch.float32))
                    else:
                        self._runtime_vh4 = None
                        self._runtime_uh4 = None
                elif (
                    _can_use_int8_gemm(dev, self.use_int8_kernel)
                    and self._runtime_uhq_t is not None
                    and self._runtime_vhq_t is not None
                    and self._runtime_uhs is not None
                    and self._runtime_vhs is not None
                ):
                    zh = _int8_linear_dynamic_act(x2d, self._runtime_vhq_t, self._runtime_vhs)
                    if self._runtime_sh is not None:
                        zh.mul_(self._runtime_sh.unsqueeze(0))
                    out2d.add_(_int8_linear_dynamic_act(zh, self._runtime_uhq_t, self._runtime_uhs))
                else:
                    zh = F.linear(x2d.to(dtype=self._runtime_vh.dtype), self._runtime_vh)
                    if self._runtime_sh is not None:
                        zh = zh * self._runtime_sh.unsqueeze(0).to(device=zh.device, dtype=zh.dtype)
                    oh = F.linear(zh, self._runtime_uh)
                    out2d.add_(oh.to(dtype=torch.float32))
            if self._runtime_vl is not None and self._runtime_ul is not None:
                if self._runtime_vl4 is not None and self._runtime_ul4 is not None:
                    zl = _run_bnb_linear4(self._runtime_vl4, x4)
                    if zl is not None:
                        if self._runtime_sl is not None:
                            zl = zl * self._runtime_sl.unsqueeze(0).to(device=zl.device, dtype=zl.dtype)
                        ol = _run_bnb_linear4(self._runtime_ul4, zl)
                    else:
                        ol = None
                    if ol is not None:
                        out2d.add_(ol.to(dtype=torch.float32))
                    else:
                        self._runtime_vl4 = None
                        self._runtime_ul4 = None
                elif (
                    _can_use_int8_gemm(dev, self.use_int8_kernel)
                    and self._runtime_ulq_t is not None
                    and self._runtime_vlq_t is not None
                    and self._runtime_uls is not None
                    and self._runtime_vls is not None
                ):
                    zl = _int8_linear_dynamic_act(x2d, self._runtime_vlq_t, self._runtime_vls)
                    if self._runtime_sl is not None:
                        zl.mul_(self._runtime_sl.unsqueeze(0))
                    out2d.add_(_int8_linear_dynamic_act(zl, self._runtime_ulq_t, self._runtime_uls))
                else:
                    zl = F.linear(x2d.to(dtype=self._runtime_vl.dtype), self._runtime_vl)
                    if self._runtime_sl is not None:
                        zl = zl * self._runtime_sl.unsqueeze(0).to(device=zl.device, dtype=zl.dtype)
                    ol = F.linear(zl, self._runtime_ul)
                    out2d.add_(ol.to(dtype=torch.float32))
            out = out2d.reshape(*x.shape[:-1], self.out_features).to(dtype=dtype)
        else:
            z = F.linear(x, self._runtime_v)
            z = z * self._runtime_sigma
            out = F.linear(z, self._runtime_u)
        if self.bias is not None:
            out = out + self.bias.to(device=dev, dtype=dtype)
        if not self.runtime_cache_persistent:
            self.clear_runtime_cache()
        return out


class Int8WeightLinear(nn.Module):
    """
    Int8 weight-only linear with per-row symmetric scales and dynamic int8 activations.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_int8_kernel: bool = True,
    ):
        super().__init__()
        self.use_int8_kernel = use_int8_kernel
        self.runtime_cache_persistent = False
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight.shape[1])
        w_q, w_s = self._q_per_row(weight)
        self.register_buffer("w_q", w_q, persistent=True)
        self.register_buffer("w_s", w_s, persistent=True)
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().clone().to(dtype=torch.float16), persistent=True)

        self._runtime_wq_t: Optional[torch.Tensor] = None
        self._runtime_ws: Optional[torch.Tensor] = None
        self._runtime_w: Optional[torch.Tensor] = None
        self._runtime_device: Optional[torch.device] = None
        self._runtime_dtype: Optional[torch.dtype] = None

    @staticmethod
    def _q_per_row(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = weight.float()
        qmax = 127.0
        s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(w / s).clamp(-qmax, qmax).to(torch.int8)
        return q.cpu(), s.squeeze(1).to(dtype=torch.float16).cpu()

    def _build_runtime_cache(self, device: torch.device, dtype: torch.dtype):
        if (
            self._runtime_wq_t is not None
            and self._runtime_ws is not None
            and self._runtime_device == device
            and self._runtime_dtype == dtype
        ):
            return
        self._runtime_wq_t = self.w_q.to(device=device).transpose(0, 1).contiguous()
        self._runtime_ws = self.w_s.to(device=device, dtype=torch.float32)
        self._runtime_w = None
        if not _can_use_int8_gemm(device, self.use_int8_kernel):
            w = self.w_q.to(device=device, dtype=torch.float32)
            s = self.w_s.to(device=device, dtype=torch.float32).unsqueeze(1)
            self._runtime_w = (w * s).to(dtype=dtype)
        self._runtime_device = device
        self._runtime_dtype = dtype

    def clear_runtime_cache(self):
        self._runtime_wq_t = None
        self._runtime_ws = None
        self._runtime_w = None
        self._runtime_device = None
        self._runtime_dtype = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        dtype = x.dtype
        self._build_runtime_cache(dev, dtype)
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        if _can_use_int8_gemm(dev, self.use_int8_kernel) and self._runtime_wq_t is not None and self._runtime_ws is not None:
            out2d = _int8_linear_dynamic_act(x2d, self._runtime_wq_t, self._runtime_ws).to(dtype=dtype)
        else:
            if self._runtime_w is None:
                w = self.w_q.to(device=dev, dtype=torch.float32)
                s = self.w_s.to(device=dev, dtype=torch.float32).unsqueeze(1)
                self._runtime_w = (w * s).to(dtype=dtype)
            out2d = F.linear(x2d.to(dtype=self._runtime_w.dtype), self._runtime_w)
        out = out2d.reshape(*x.shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias.to(device=dev, dtype=dtype)
        if not self.runtime_cache_persistent:
            self.clear_runtime_cache()
        return out


class Int8Embedding(nn.Module):
    """
    Int8 weight-only embedding with per-row symmetric scales.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        padding_idx: Optional[int] = None,
        output_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_embeddings = int(weight.shape[0])
        self.embedding_dim = int(weight.shape[1])
        self.padding_idx = padding_idx
        self.output_dtype = output_dtype
        self.runtime_cache_persistent = False
        emb_q, emb_s = self._q_per_row(weight)
        self.register_buffer("emb_q", emb_q, persistent=True)
        self.register_buffer("emb_s", emb_s, persistent=True)

    @staticmethod
    def _q_per_row(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = weight.float()
        qmax = 127.0
        s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(w / s).clamp(-qmax, qmax).to(torch.int8)
        return q.cpu(), s.squeeze(1).to(dtype=torch.float16).cpu()

    def clear_runtime_cache(self):
        return

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        dev = input_ids.device
        flat = input_ids.reshape(-1)
        q_rows = self.emb_q.to(device=dev).index_select(0, flat)
        s_rows = self.emb_s.to(device=dev, dtype=torch.float32).index_select(0, flat)
        out = q_rows.to(dtype=torch.float32) * s_rows.unsqueeze(1)
        out = out.to(dtype=self.output_dtype)
        return out.reshape(*input_ids.shape, self.embedding_dim)


def apply_two_path_quantization(
    model: nn.Module,
    pairs: List[PairModules],
    alloc: Dict[str, torch.Tensor],
    high_bit: int = 8,
    low_bit: int = 4,
    explicit_sigma: bool = False,
    sigma_eps: float = 1e-12,
    use_int8_kernel: bool = True,
    use_int4_kernel: bool = False,
    int4_quant_type: str = "nf4",
):
    if use_int4_kernel and bnb is None:
        print("Warning: --mp-enable-int4-kernel is set but bitsandbytes is not available. Falling back to non-int4 kernels.")
    for pair in pairs:
        if pair.key not in alloc:
            continue
        alloc_entry = alloc[pair.key]
        if alloc_entry.dtype == torch.bool:
            high_mask = alloc_entry
            rank = high_mask.numel()
            high_idx = torch.where(high_mask)[0]
            low_idx = torch.where(~high_mask)[0]
            if high_idx.numel() == 0 and low_idx.numel() == 0 and rank > 0:
                low_idx = torch.arange(rank)
        else:
            states = alloc_entry.to(dtype=torch.uint8)
            high_idx = torch.where(states == MP_STATE_HIGH)[0]
            low_idx = torch.where(states == MP_STATE_LOW)[0]

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
                use_int8_kernel=use_int8_kernel,
                use_int4_kernel=use_int4_kernel,
                int4_quant_type=int4_quant_type,
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
                use_int8_kernel=use_int8_kernel,
                use_int4_kernel=use_int4_kernel,
                int4_quant_type=int4_quant_type,
            )
        parent = _get_submodule(model, pair.parent_path)
        setattr(parent, f"{pair.stem}_mp_proj", mp)


@torch.no_grad()
def apply_non_svd_int8_quantization(
    model: nn.Module,
    pairs: List[PairModules],
    use_int8_kernel: bool = True,
    exclude_lm_head: bool = False,
) -> Dict[str, int]:
    """
    Quantize non-SVD nn.Linear modules to int8 weight-only.
    Excludes all *_u_proj/*_v_proj pair modules tracked by `pairs`.
    """
    excluded_ids = set()
    for pair in pairs:
        excluded_ids.add(id(pair.u_module))
        excluded_ids.add(id(pair.v_module))

    replaced = 0
    skipped = 0
    total_linear = 0
    named = list(model.named_modules())
    for path, mod in named:
        if path == "" or not isinstance(mod, nn.Linear):
            continue
        total_linear += 1
        if id(mod) in excluded_ids:
            skipped += 1
            continue
        if path.endswith("_u_proj") or path.endswith("_v_proj"):
            skipped += 1
            continue
        if path.endswith("_mp_proj"):
            skipped += 1
            continue
        if exclude_lm_head and path == "lm_head":
            skipped += 1
            continue
        parent_path, name = _split_parent_and_name(path)
        parent = _get_submodule(model, parent_path)
        bias = mod.bias.data if mod.bias is not None else None
        qmod = Int8WeightLinear(
            weight=mod.weight.data,
            bias=bias,
            use_int8_kernel=use_int8_kernel,
        )
        setattr(parent, name, qmod)
        replaced += 1

    return {
        "total_linear": total_linear,
        "replaced": replaced,
        "skipped": skipped,
    }


@torch.no_grad()
def apply_embed_tokens_int8_quantization(
    model: nn.Module,
    include_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Quantize embedding modules named 'embed_tokens' (or custom names) to int8 weight-only.
    """
    include = include_names if include_names is not None else ["embed_tokens"]
    replaced = 0
    skipped = 0
    total_embedding = 0
    for path, mod in list(model.named_modules()):
        if path == "" or not isinstance(mod, nn.Embedding):
            continue
        total_embedding += 1
        leaf = path.rsplit(".", 1)[-1]
        if leaf not in include:
            skipped += 1
            continue
        parent_path, name = _split_parent_and_name(path)
        parent = _get_submodule(model, parent_path)
        out_dtype = mod.weight.dtype if mod.weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
        qmod = Int8Embedding(
            weight=mod.weight.data,
            padding_idx=mod.padding_idx,
            output_dtype=out_dtype,
        )
        setattr(parent, name, qmod)
        replaced += 1
    return {
        "total_embedding": total_embedding,
        "replaced": replaced,
        "skipped": skipped,
    }


@torch.no_grad()
def clear_mixed_precision_runtime_cache(model: nn.Module) -> int:
    """
    Clear transient runtime caches in MP quantized modules so saved checkpoints
    only contain persistent quantized weights/buffers.
    Returns the number of modules cleared.
    """
    cleared = 0
    for mod in model.modules():
        if isinstance(mod, (TwoPathLowRankLinear, TwoPathSigmaLowRankLinear, Int8WeightLinear, Int8Embedding)):
            mod.clear_runtime_cache()
            cleared += 1
    return cleared


@torch.no_grad()
def set_mixed_precision_runtime_cache_policy(model: nn.Module, persistent: bool) -> int:
    """
    Configure whether MP modules keep runtime caches across calls.
    persistent=False reduces peak GPU memory; persistent=True is faster.
    Returns number of modules updated.
    """
    touched = 0
    for mod in model.modules():
        if isinstance(mod, (TwoPathLowRankLinear, TwoPathSigmaLowRankLinear, Int8WeightLinear, Int8Embedding)):
            mod.runtime_cache_persistent = bool(persistent)
            if not persistent:
                mod.clear_runtime_cache()
            touched += 1
    return touched


@torch.no_grad()
def strip_original_low_rank_weights(
    model: nn.Module,
    pairs: List[PairModules],
) -> Dict[str, int]:
    """
    Strip original *_u_proj/*_v_proj modules after mp_proj is installed.
    This reduces checkpoint size by removing redundant full-precision low-rank params.
    """
    removed_params = 0
    replaced_modules = 0
    skipped_pairs = 0

    for pair in pairs:
        parent = _get_submodule(model, pair.parent_path)
        mp_name = f"{pair.stem}_mp_proj"
        if not hasattr(parent, mp_name) or getattr(parent, mp_name) is None:
            skipped_pairs += 1
            continue

        if hasattr(parent, pair.u_name):
            u_mod = getattr(parent, pair.u_name)
            if isinstance(u_mod, nn.Module) and not isinstance(u_mod, nn.Identity):
                for param in u_mod.parameters(recurse=True):
                    removed_params += int(param.numel())
                setattr(parent, pair.u_name, nn.Identity())
                replaced_modules += 1

        if hasattr(parent, pair.v_name):
            v_mod = getattr(parent, pair.v_name)
            if isinstance(v_mod, nn.Module) and not isinstance(v_mod, nn.Identity):
                for param in v_mod.parameters(recurse=True):
                    removed_params += int(param.numel())
                setattr(parent, pair.v_name, nn.Identity())
                replaced_modules += 1

    return {
        "removed_params": removed_params,
        "replaced_modules": replaced_modules,
        "skipped_pairs": skipped_pairs,
    }
