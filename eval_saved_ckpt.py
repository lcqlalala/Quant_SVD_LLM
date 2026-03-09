import argparse
import os

import torch

from evaluater import ppl_eval
from utils.model_utils import get_model_from_local


def inspect_checkpoint(path: str) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"checkpoint_path: {path}")
    print(f"checkpoint_type: {type(obj).__name__}")
    if isinstance(obj, dict):
        print(f"top_level_keys: {sorted(list(obj.keys()))}")
        fmt = obj.get("format", None)
        if fmt is not None:
            print(f"format: {fmt}")
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
            print(f"state_dict_keys: {len(sd)}")
            packed_n = sum(1 for k in sd.keys() if k.endswith("_packed4"))
            q_int8_n = sum(1 for k, v in sd.items() if k.endswith((".ul_q", ".vl_q")) and isinstance(v, torch.Tensor) and v.dtype == torch.int8)
            print(f"packed4_tensors: {packed_n}")
            print(f"raw_lowbit_int8_tensors: {q_int8_n}")
        if "packed_lowbit_q" in obj:
            print(f"packed_lowbit_q: {obj['packed_lowbit_q']}")
        if "packed_lowbit_q_stats" in obj:
            print(f"packed_lowbit_q_stats: {obj['packed_lowbit_q_stats']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved checkpoint (.pt)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path for loading")
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--model_dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inspect_only", action="store_true", help="Only inspect checkpoint without evaluation")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")

    inspect_checkpoint(args.model_path)
    if args.inspect_only:
        return

    model, tokenizer = get_model_from_local(args.model_path, tokenizer_path=args.tokenizer_path)
    if args.model_dtype == "fp16":
        model = model.half()
    elif args.model_dtype == "bf16":
        model = model.bfloat16()
    else:
        model = model.float()
    model = model.to(args.device)
    model.eval()

    ppl_eval(
        model,
        tokenizer,
        datasets=[args.dataset],
        model_seq_len=args.model_seq_len,
        batch_size=args.eval_batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

