import argparse
import os
import inspect

import torch

from evaluater import ppl_eval
from utils.model_utils import get_model_from_local
from utils.mixed_precision import set_mixed_precision_runtime_cache_policy


def _mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def _tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _module_size_report(model: torch.nn.Module) -> dict:
    param_count = 0
    trainable_param_count = 0
    param_bytes = 0
    buffer_count = 0
    buffer_bytes = 0
    for p in model.parameters():
        param_count += int(p.numel())
        param_bytes += _tensor_bytes(p)
        if p.requires_grad:
            trainable_param_count += int(p.numel())
    for b in model.buffers():
        buffer_count += int(b.numel())
        buffer_bytes += _tensor_bytes(b)
    return {
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "buffer_count": buffer_count,
        "param_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "total_bytes": param_bytes + buffer_bytes,
    }


def _print_module_size_report(model: torch.nn.Module, prefix: str = "Model") -> None:
    report = _module_size_report(model)
    print(
        f"{prefix} params: total={report['param_count']:,}, "
        f"trainable={report['trainable_param_count']:,}, "
        f"bytes={_mib(report['param_bytes']):.2f} MiB"
    )
    print(
        f"{prefix} buffers: total={report['buffer_count']:,}, "
        f"bytes={_mib(report['buffer_bytes']):.2f} MiB"
    )
    print(f"{prefix} params+buffers memory: {_mib(report['total_bytes']):.2f} MiB")


def _print_cuda_memory(device: str, prefix: str) -> None:
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    torch.cuda.synchronize(dev)
    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    peak_allocated = torch.cuda.max_memory_allocated(dev)
    peak_reserved = torch.cuda.max_memory_reserved(dev)
    print(
        f"{prefix} CUDA memory: "
        f"allocated={_mib(allocated):.2f} MiB, "
        f"reserved={_mib(reserved):.2f} MiB, "
        f"peak_allocated={_mib(peak_allocated):.2f} MiB, "
        f"peak_reserved={_mib(peak_reserved):.2f} MiB"
    )


def inspect_checkpoint(path: str) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"checkpoint_path: {path}")
    print(f"checkpoint_type: {type(obj).__name__}")
    if isinstance(obj, dict):
        print(f"top_level_keys: {sorted(list(obj.keys()))}")
        fmt = obj.get("format", None)
        if fmt is not None:
            print(f"format: {fmt}")
        if "arch_model_path" in obj:
            print(f"arch_model_path: {obj.get('arch_model_path')}")
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
            print(f"state_dict_keys: {len(sd)}")
            packed_n = sum(1 for k in sd.keys() if k.endswith("_packed4"))
            q_int8_n = sum(1 for k, v in sd.items() if k.endswith((".ul_q", ".vl_q")) and isinstance(v, torch.Tensor) and v.dtype == torch.int8)
            nonsvd_int8_n = sum(1 for k, v in sd.items() if k.endswith(".w_q") and isinstance(v, torch.Tensor) and v.dtype == torch.int8)
            embed_int8_n = sum(1 for k, v in sd.items() if k.endswith(".emb_q") and isinstance(v, torch.Tensor) and v.dtype == torch.int8)
            print(f"packed4_tensors: {packed_n}")
            print(f"raw_lowbit_int8_tensors: {q_int8_n}")
            print(f"nonsvd_int8_tensors: {nonsvd_int8_n}")
            print(f"embed_int8_tensors: {embed_int8_n}")
        if "packed_lowbit_q" in obj:
            print(f"packed_lowbit_q: {obj['packed_lowbit_q']}")
        if "packed_lowbit_q_stats" in obj:
            print(f"packed_lowbit_q_stats: {obj['packed_lowbit_q_stats']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved checkpoint (.pt)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path for loading")
    parser.add_argument(
        "--arch_model_path",
        type=str,
        default=None,
        help="Original SVD template checkpoint path (required when state_dict cannot rebuild MP modules from base model).",
    )
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--model_dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--mp-persistent-runtime-cache",
        action="store_true",
        help="Keep MP dequantized/runtime weights cached during evaluation for faster inference.",
    )
    parser.add_argument("--inspect_only", action="store_true", help="Only inspect checkpoint without evaluation")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")

    inspect_checkpoint(args.model_path)
    if args.inspect_only:
        return

    sig = inspect.signature(get_model_from_local)
    if "arch_model_path" in sig.parameters:
        model, tokenizer = get_model_from_local(
            args.model_path,
            tokenizer_path=args.tokenizer_path,
            arch_model_path=args.arch_model_path,
        )
    else:
        if args.arch_model_path:
            print(
                "Warning: current utils/model_utils.py does not support --arch_model_path. "
                "Ignoring it; please sync latest model_utils.py to avoid architecture mismatch."
            )
        model, tokenizer = get_model_from_local(
            args.model_path,
            tokenizer_path=args.tokenizer_path,
        )
    if args.model_dtype == "fp16":
        model = model.half()
    elif args.model_dtype == "bf16":
        model = model.bfloat16()
    else:
        model = model.float()
    model = model.to(args.device)
    model.eval()
    if args.mp_persistent_runtime_cache:
        touched = set_mixed_precision_runtime_cache_policy(model, persistent=True)
        print(f"Configured MP runtime cache policy on {touched} modules: persistent=True")

    _print_module_size_report(model, prefix="Loaded model")
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        _print_cuda_memory(args.device, "Before eval")

    ppls, throughput = ppl_eval(
        model,
        tokenizer,
        datasets=[args.dataset],
        model_seq_len=args.model_seq_len,
        batch_size=args.eval_batch_size,
        device=args.device,
    )
    print(f"Eval PPL summary: {ppls}")
    print(f"Eval throughput summary (tokens/s): {throughput}")
    _print_cuda_memory(args.device, "After eval")


if __name__ == "__main__":
    main()
