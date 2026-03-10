#coding:utf8
import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Tuple

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# bandaid fix
dev = torch.device("cuda")

def get_model_from_huggingface(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    if "opt" in model_id or "mistral" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None)
    model.seqlen = 2048
    return model, tokenizer

def _resolve_tokenizer_source(model_path: str, tokenizer_path: Optional[str], model_obj) -> Optional[str]:
    if tokenizer_path and not tokenizer_path.endswith(".pt"):
        return tokenizer_path
    if os.path.isdir(model_path):
        return model_path
    if hasattr(model_obj, "config") and hasattr(model_obj.config, "_name_or_path"):
        name_or_path = getattr(model_obj.config, "_name_or_path", None)
        if isinstance(name_or_path, str) and len(name_or_path) > 0:
            return name_or_path
    parent_dir = os.path.dirname(model_path)
    if os.path.exists(os.path.join(parent_dir, "tokenizer.json")) or os.path.exists(os.path.join(parent_dir, "tokenizer.model")):
        return parent_dir
    return None


def _load_tokenizer(tokenizer_source: Optional[str]):
    if tokenizer_source is None:
        return None
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except Exception:
        return None


def _load_model_object_from_checkpoint(checkpoint_path: str) -> Optional[nn.Module]:
    """
    Best-effort loader for a full model object checkpoint.
    Returns None when the checkpoint does not contain a direct model object.
    """
    if not isinstance(checkpoint_path, str) or not os.path.isfile(checkpoint_path):
        return None
    try:
        obj = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    except Exception:
        return None
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], nn.Module):
            return obj["model"]
        if "module" in obj and isinstance(obj["module"], nn.Module):
            return obj["module"]
    return None


def _get_submodule_by_path(model: nn.Module, path: str) -> nn.Module:
    if path == "":
        return model
    cur = model
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _unpack_int4_signed_rows(packed: torch.Tensor, orig_cols: int) -> torch.Tensor:
    if packed.ndim != 2:
        raise ValueError(f"Expected 2D packed tensor, got shape={tuple(packed.shape)}")
    p = packed.to(dtype=torch.uint8)
    lo = (p & 0x0F).to(torch.int16)
    hi = ((p >> 4) & 0x0F).to(torch.int16)
    rows = p.shape[0]
    out = torch.empty((rows, p.shape[1] * 2), dtype=torch.int16, device=p.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    out = out[:, :orig_cols]
    return (out - 8).to(torch.int8)


def _unpack_lowbit_q_state_dict(state_dict: dict) -> dict:
    """
    Restore packed low-bit q tensors saved as:
      <key>_packed4 (uint8), <key>_orig_cols (int32 scalar tensor)
    back to:
      <key> (int8)
    """
    out = {}
    packed_suffix = "_packed4"
    cols_suffix = "_orig_cols"

    packed_bases = set()
    for k in state_dict.keys():
        if k.endswith(packed_suffix):
            packed_bases.add(k[: -len(packed_suffix)])

    for k, v in state_dict.items():
        if k.endswith(packed_suffix) or k.endswith(cols_suffix):
            continue
        out[k] = v

    for base in sorted(packed_bases):
        packed_key = f"{base}{packed_suffix}"
        cols_key = f"{base}{cols_suffix}"
        packed = state_dict[packed_key]
        if cols_key in state_dict and isinstance(state_dict[cols_key], torch.Tensor):
            orig_cols = int(state_dict[cols_key].view(-1)[0].item())
        else:
            orig_cols = int(packed.shape[1] * 2)
        out[base] = _unpack_int4_signed_rows(packed, orig_cols)

    return out


def _install_mp_modules_from_state_dict(
    model: nn.Module,
    state_dict: dict,
    mp_config: Optional[dict] = None,
) -> int:
    from utils.mixed_precision import TwoPathLowRankLinear, TwoPathSigmaLowRankLinear

    use_int8_kernel = True
    use_int4_kernel = False
    int4_quant_type = "nf4"
    if isinstance(mp_config, dict):
        use_int8_kernel = bool(mp_config.get("use_int8_kernel", True))
        use_int4_kernel = bool(mp_config.get("use_int4_kernel", False))
        int4_quant_type = str(mp_config.get("int4_quant_type", "nf4"))

    prefixes = set()
    for key in state_dict.keys():
        marker = "_mp_proj."
        if marker not in key:
            continue
        prefix = key.split(marker)[0] + "_mp_proj"
        prefixes.add(prefix)

    installed = 0
    stripped_uv = 0

    def _rank_from_q(prefix: str, q_name: str, s_name: str) -> int:
        q_key = f"{prefix}.{q_name}"
        s_key = f"{prefix}.{s_name}"
        if q_key in state_dict and isinstance(state_dict[q_key], torch.Tensor):
            q = state_dict[q_key]
            if q.numel() == 0:
                return 0
            if q.ndim >= 2:
                return int(q.shape[1])
            if s_key in state_dict and isinstance(state_dict[s_key], torch.Tensor):
                s = state_dict[s_key]
                if s.numel() > 0:
                    return int(s.numel())
            return 0
        if s_key in state_dict and isinstance(state_dict[s_key], torch.Tensor):
            s = state_dict[s_key]
            if s.numel() > 0:
                return int(s.numel())
        return 0

    for prefix in sorted(prefixes):
        parent_path, mp_name = prefix.rsplit(".", 1)
        if not mp_name.endswith("_mp_proj"):
            continue
        stem = mp_name[: -len("_mp_proj")]
        parent = _get_submodule_by_path(model, parent_path)
        u_name = f"{stem}_u_proj"
        v_name = f"{stem}_v_proj"
        if not hasattr(parent, u_name) or not hasattr(parent, v_name):
            continue
        u_mod = getattr(parent, u_name)
        v_mod = getattr(parent, v_name)
        if not isinstance(u_mod, nn.Linear) or not isinstance(v_mod, nn.Linear):
            continue

        nh = _rank_from_q(prefix, "uh_q", "uh_s")
        nl = _rank_from_q(prefix, "ul_q", "ul_s")
        high_idx = torch.arange(nh, dtype=torch.long)
        low_idx = torch.arange(nh, nh + nl, dtype=torch.long)

        is_sigma = (f"{prefix}.sh" in state_dict) or (f"{prefix}.sl" in state_dict)
        if is_sigma:
            from utils.mixed_precision import _decompose_uv_to_explicit_sigma

            u_basis, sigma_vec, v_basis, _ = _decompose_uv_to_explicit_sigma(
                u_weight=u_mod.weight.data.detach().float().cpu(),
                v_weight=v_mod.weight.data.detach().float().cpu(),
                eps=1e-12,
            )
            mp = TwoPathSigmaLowRankLinear(
                u_basis=u_basis,
                v_basis=v_basis,
                sigma=sigma_vec,
                bias=u_mod.bias.data if u_mod.bias is not None else None,
                high_idx=high_idx,
                low_idx=low_idx,
                high_bit=8,
                low_bit=4,
                use_int8_kernel=use_int8_kernel,
                use_int4_kernel=use_int4_kernel,
                int4_quant_type=int4_quant_type,
            )
        else:
            mp = TwoPathLowRankLinear(
                u_weight=u_mod.weight.data,
                v_weight=v_mod.weight.data,
                bias=u_mod.bias.data if u_mod.bias is not None else None,
                high_idx=high_idx,
                low_idx=low_idx,
                high_bit=8,
                low_bit=4,
                use_int8_kernel=use_int8_kernel,
                use_int4_kernel=use_int4_kernel,
                int4_quant_type=int4_quant_type,
            )
        setattr(parent, mp_name, mp)
        # For MP checkpoints, original low-rank u/v modules are redundant and not saved.
        # Replacing them with Identity avoids load-time missing keys and reduces memory.
        if hasattr(parent, u_name):
            setattr(parent, u_name, nn.Identity())
            stripped_uv += 1
        if hasattr(parent, v_name):
            setattr(parent, v_name, nn.Identity())
            stripped_uv += 1
        installed += 1
    if stripped_uv > 0:
        print(f"Stripped original u/v modules while installing MP: {stripped_uv}")
    return installed


def _install_non_svd_int8_modules_from_state_dict(
    model: nn.Module,
    state_dict: dict,
    mp_config: Optional[dict] = None,
) -> int:
    """
    Install Int8WeightLinear modules for non-SVD linears saved as:
      <module_path>.w_q / <module_path>.w_s
    """
    from utils.mixed_precision import Int8WeightLinear

    use_int8_kernel = True
    if isinstance(mp_config, dict):
        use_int8_kernel = bool(mp_config.get("use_int8_kernel", True))

    prefixes = set()
    for key in state_dict.keys():
        if key.endswith(".w_q"):
            prefixes.add(key[: -len(".w_q")])

    installed = 0
    for prefix in sorted(prefixes):
        q_key = f"{prefix}.w_q"
        s_key = f"{prefix}.w_s"
        if q_key not in state_dict or s_key not in state_dict:
            continue
        q = state_dict[q_key]
        s = state_dict[s_key]
        if not isinstance(q, torch.Tensor) or not isinstance(s, torch.Tensor):
            continue
        if q.ndim != 2:
            continue

        if "." in prefix:
            parent_path, name = prefix.rsplit(".", 1)
        else:
            parent_path, name = "", prefix
        parent = _get_submodule_by_path(model, parent_path)
        if not hasattr(parent, name):
            continue
        mod = getattr(parent, name)
        if isinstance(mod, Int8WeightLinear):
            continue
        if not isinstance(mod, nn.Linear):
            continue

        bias = mod.bias.data if mod.bias is not None else None
        qmod = Int8WeightLinear(
            weight=mod.weight.data,
            bias=bias,
            use_int8_kernel=use_int8_kernel,
        )
        setattr(parent, name, qmod)
        installed += 1
    return installed


def get_model_from_local(
    model_id: str,
    tokenizer_path: Optional[str] = None,
    arch_model_path: Optional[str] = None,
) -> Tuple[nn.Module, object]:
    # HF local directory path
    if os.path.isdir(model_id):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = _load_tokenizer(tokenizer_path or model_id)
        if tokenizer is None:
            raise ValueError(
                f"Failed to load tokenizer from '{tokenizer_path or model_id}'. "
                "Please pass a valid tokenizer path via --tokenizer_path."
            )
        model.seqlen = 2048
        return model, tokenizer

    obj = torch.load(model_id, weights_only=False, map_location='cpu')
    model = None
    tokenizer = None

    if isinstance(obj, dict):
        if "model" in obj:
            model = obj["model"]
        elif obj.get("format", None) == "svd_mp_state_dict_v1" and isinstance(obj.get("state_dict", None), dict):
            from transformers import AutoModelForCausalLM

            state_dict = obj["state_dict"]
            if bool(obj.get("packed_lowbit_q", False)):
                state_dict = _unpack_lowbit_q_state_dict(state_dict)

            # Prefer loading from an SVD template checkpoint (same architecture as the saved state_dict).
            # Falling back to HF base model can silently fail to install MP modules.
            model = None
            tried_arch_paths = []
            for candidate in [arch_model_path, obj.get("arch_model_path", None), obj.get("svd_template_path", None)]:
                if not isinstance(candidate, str) or len(candidate) == 0:
                    continue
                if os.path.abspath(candidate) == os.path.abspath(model_id):
                    continue
                tried_arch_paths.append(candidate)
                model = _load_model_object_from_checkpoint(candidate)
                if model is not None:
                    print(f"Loaded SVD template architecture from: {candidate}")
                    break

            if model is None:
                base_model_path = obj.get("base_model_path", None)
                if base_model_path is None:
                    raise ValueError(
                        "State-dict checkpoint requires a base architecture. "
                        "Provide --arch_model_path (recommended, SVD template checkpoint) "
                        "or save checkpoint with base_model_path."
                    )
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map="cpu",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                if len(tried_arch_paths) > 0:
                    print(
                        "Warning: failed to load SVD template architecture from provided paths; "
                        f"falling back to base model: {base_model_path}"
                    )

            installed = _install_mp_modules_from_state_dict(model, state_dict, obj.get("mp_config", None))
            installed_non_svd = _install_non_svd_int8_modules_from_state_dict(model, state_dict, obj.get("mp_config", None))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if installed > 0:
                print(f"Installed MP modules from checkpoint: {installed}")
            if installed_non_svd > 0:
                print(f"Installed non-SVD int8 modules from checkpoint: {installed_non_svd}")
            if len(unexpected) > 0:
                print(f"Warning: unexpected keys while loading state_dict: {len(unexpected)}")
            if len(missing) > 0:
                print(f"Warning: missing keys while loading state_dict: {len(missing)}")
            has_mp_keys = any("_mp_proj." in k for k in state_dict.keys())
            if has_mp_keys and installed == 0:
                raise ValueError(
                    "Detected MP state_dict keys but installed 0 MP modules. "
                    "This means architecture mismatch (likely loading dense HF model instead of SVD template). "
                    "Please pass --arch_model_path to the original SVD checkpoint used before quantization."
                )
        elif isinstance(obj.get("state_dict", None), dict):
            raise ValueError(
                "Checkpoint only contains 'state_dict'. This repo requires a full model object checkpoint "
                "or a HuggingFace model directory."
            )
        elif "module" in obj and isinstance(obj["module"], nn.Module):
            model = obj["module"]

        tokenizer = obj.get("tokenizer", None)
        if tokenizer is None and obj.get("format", None) == "svd_mp_state_dict_v1":
            tok_source = tokenizer_path
            if isinstance(tok_source, str) and tok_source.endswith(".pt"):
                tok_source = None
            tokenizer = _load_tokenizer(tok_source or obj.get("tokenizer_path", None) or obj.get("base_model_path", None))
    elif isinstance(obj, nn.Module):
        model = obj

    if model is None:
        raise ValueError(
            f"Unsupported checkpoint format at '{model_id}'. Expected keys like 'model' (and optional 'tokenizer')."
        )

    if tokenizer is None:
        tokenizer_source = _resolve_tokenizer_source(model_id, tokenizer_path, model)
        tokenizer = _load_tokenizer(tokenizer_source)
        if tokenizer is None:
            raise ValueError(
                "Tokenizer is missing in checkpoint and automatic loading failed. "
                "Please pass --tokenizer_path to a valid HuggingFace tokenizer/model directory or repo id."
            )

    if not hasattr(model, "seqlen"):
        model.seqlen = 2048
    return model, tokenizer

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
