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
        elif isinstance(obj.get("state_dict", None), dict):
            raise ValueError(
                "Checkpoint only contains 'state_dict'. This repo requires a full model object checkpoint "
                "or a HuggingFace model directory."
            )
        elif "module" in obj and isinstance(obj["module"], nn.Module):
            model = obj["module"]

        tokenizer = obj.get("tokenizer", None)
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
