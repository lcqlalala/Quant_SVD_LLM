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
