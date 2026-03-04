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
    if tokenizer_path:
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


def _get_submodule_by_path(model: nn.Module, path: str) -> nn.Module:
    if path == "":
        return model
    cur = model
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _install_mp_modules_from_state_dict(
    model: nn.Module,
    state_dict: dict,
    mp_config: Optional[dict] = None,
):
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

        uh_key = f"{prefix}.uh_q"
        ul_key = f"{prefix}.ul_q"
        nh = int(state_dict[uh_key].shape[1]) if uh_key in state_dict else 0
        nl = int(state_dict[ul_key].shape[1]) if ul_key in state_dict else 0
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


def get_model_from_local(model_id: str, tokenizer_path: Optional[str] = None) -> Tuple[nn.Module, object]:
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

            base_model_path = tokenizer_path or obj.get("base_model_path", None)
            if base_model_path is None:
                raise ValueError(
                    "State-dict checkpoint requires base model path to rebuild architecture. "
                    "Please pass --tokenizer_path (or save checkpoint with base_model_path)."
                )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="cpu",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            _install_mp_modules_from_state_dict(model, obj["state_dict"], obj.get("mp_config", None))
            missing, unexpected = model.load_state_dict(obj["state_dict"], strict=False)
            if len(unexpected) > 0:
                print(f"Warning: unexpected keys while loading state_dict: {len(unexpected)}")
            if len(missing) > 0:
                print(f"Warning: missing keys while loading state_dict: {len(missing)}")
        elif isinstance(obj.get("state_dict", None), dict):
            raise ValueError(
                "Checkpoint only contains 'state_dict'. This repo requires a full model object checkpoint "
                "or a HuggingFace model directory."
            )
        elif "module" in obj and isinstance(obj["module"], nn.Module):
            model = obj["module"]

        tokenizer = obj.get("tokenizer", None)
        if tokenizer is None and obj.get("format", None) == "svd_mp_state_dict_v1":
            tokenizer = _load_tokenizer(tokenizer_path or obj.get("tokenizer_path", None) or obj.get("base_model_path", None))
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
