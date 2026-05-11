# coding:utf8
"""
Compatibility launcher for running the original SVDLLM.py with the
LLaMA-3/3.1 GQA-aware SVD component.

Usage is the same as SVDLLM.py, but launch this file instead:

    python SVDLLM_llama3_wrapper.py <original SVDLLM.py args>

The wrapper must install component.svd_llama3_1 under the legacy module name
component.svd_llama before SVDLLM.py is imported.  This lets the original SVD
pipeline reuse its existing imports while constructing GQA-correct K/V modules.
"""

import importlib
import runpy
import sys

# Transformers 4.43.0 has an eager-default bug in LlamaRotaryEmbedding:
#   config.rope_scaling.get("rope_type", config.rope_scaling["type"])
# The default expression is evaluated before get(), so LLaMA-3.1 configs that
# only contain "rope_type" can still raise KeyError("type").  Normalize the
# config right after construction and before AutoModelForCausalLM builds layers.
try:
    from transformers.models.llama.configuration_llama import LlamaConfig

    _orig_llama_config_init = LlamaConfig.__init__

    def _patched_llama_config_init(self, *args, **kwargs):
        _orig_llama_config_init(self, *args, **kwargs)
        rope_scaling = getattr(self, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            if "type" not in rope_scaling and "rope_type" in rope_scaling:
                rope_scaling["type"] = rope_scaling["rope_type"]
            if "rope_type" not in rope_scaling and "type" in rope_scaling:
                rope_scaling["rope_type"] = rope_scaling["type"]

    LlamaConfig.__init__ = _patched_llama_config_init
except Exception as exc:
    print(f"Warning: failed to install LLaMA-3.1 rope_scaling compatibility patch: {exc}")

# 1) Import LLaMA-3/3.1 GQA-aware two-factor implementation.
_fixed_component = importlib.import_module("component.svd_llama3_1")

# 2) Make original `from component.svd_llama import ...` resolve to it.
sys.modules["component.svd_llama"] = _fixed_component

# 3) Provide legacy class names expected by SVDLLM.py.
if not hasattr(_fixed_component, "SVD_LlamaAttention"):
    _fixed_component.SVD_LlamaAttention = _fixed_component.SVD_Llama3Attention
if not hasattr(_fixed_component, "SVD_LlamaMLP"):
    _fixed_component.SVD_LlamaMLP = _fixed_component.SVD_Llama3MLP

# 4) Run original main script with unchanged CLI arguments.
runpy.run_module("SVDLLM", run_name="__main__")
