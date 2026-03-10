import torch
import numpy as np
from tqdm import tqdm
import time
import itertools
from utils.data_utils import get_test_data
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


def _sync_if_cuda(device):
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _model_weight_memory_bytes(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


@torch.no_grad()
def ppl_eval(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()
    ppls = {}
    throughput = {}
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size = batch_size)
        nll_sum = 0.0
        nll_count = 0
        total_tokens = 0
        total_target_tokens = 0
        total_forward_time = 0.0
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            output = model(batch, use_cache=False)
            _sync_if_cuda(device)
            total_forward_time += (time.perf_counter() - t0)
            total_tokens += int(batch.numel())
            total_target_tokens += int(batch[:, 1:].numel())
            lm_logits = output.logits
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                loss_sum = loss_fct(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                nll_sum += float(loss_sum.item())
                nll_count += int(shift_labels.numel())
            del output, lm_logits
        if nll_count == 0:
            ppl = float("inf")
        else:
            ppl = float(np.exp(nll_sum / max(nll_count, 1)))
        ppls[dataset] = ppl
        tps_all = total_tokens / max(total_forward_time, 1e-12)
        tps_target = total_target_tokens / max(total_forward_time, 1e-12)
        throughput[dataset] = {
            "tokens_per_sec_all": tps_all,
            "tokens_per_sec_target": tps_target,
            "forward_time_sec": total_forward_time,
        }
        print(
            f"[{dataset}] throughput: "
            f"all_tokens/s={tps_all:.2f}, target_tokens/s={tps_target:.2f}, "
            f"forward_time={total_forward_time:.2f}s"
        )
    print("PPL after pruning: {}".format(ppls))
    print("Throughput (tokens/s): {}".format(throughput))
    weight_mib = _model_weight_memory_bytes(model) / (1024.0 * 1024.0)
    if torch.device(device).type == "cuda":
        alloc_mib = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        reserved_mib = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        print("Weight Memory (params+buffers): {:.2f} MiB".format(weight_mib))
        print("CUDA Allocated Memory: {:.2f} MiB".format(alloc_mib))
        print("CUDA Reserved Memory: {:.2f} MiB\n".format(reserved_mib))
    else:
        print("Weight Memory (params+buffers): {:.2f} MiB\n".format(weight_mib))

# only call this function when for 65b or more model    
@torch.no_grad()
def ppl_eval_large(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], seq_len=2048, batch_size=32, device="cuda"):
    import  torch.nn as nn
    class LlamaRMSNorm(nn.Module):
        def __init__(self, hidden_size=model.config.hidden_size, eps=model.config.rms_norm_eps):
            """
            LlamaRMSNorm is equivalent to T5LayerNorm
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
    norm = LlamaRMSNorm().half().cuda()
    lm_head = model.lm_head.cuda()
    model.eval()
    ppls = {}
    throughput = {}
    layers = model.model.layers
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        nll_sum = 0.0
        nll_count = 0
        total_tokens = 0
        total_target_tokens = 0
        total_forward_time = 0.0
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        for batch in tqdm(test_loader):
            _sync_if_cuda("cuda")
            t0 = time.perf_counter()
            model.model.embed_tokens = model.model.embed_tokens.cuda()
            model.model.norm = model.model.norm.cuda()
            layers[0] = layers[0].cuda()

            dtype = next(iter(model.parameters())).dtype
            inps = torch.zeros(
                (batch.shape[0], model.seqlen, model.config.hidden_size), dtype=dtype, device="cuda"
            )
            cache = {'i': 0, 'attention_mask': None, "position_ids": None}
            class Catcher(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, **kwargs):
                    inps[cache['i']] = inp
                    cache['i'] += 1
                    if cache['attention_mask'] is None:
                        cache['attention_mask'] = kwargs['attention_mask']
                        cache['position_ids'] = kwargs['position_ids']
                    else:
                        cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                        cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
                    raise ValueError
            layers[0] = Catcher(layers[0])
            for j in range(batch.shape[0]):
                try:
                    model(batch[j].unsqueeze(0).cuda())
                except ValueError:
                    pass
            layers[0] = layers[0].module
            layers[0] = layers[0].cpu()
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            model.model.norm = model.model.norm.cpu()
            torch.cuda.empty_cache()
            attention_masks = cache['attention_mask']
            position_ids = cache['position_ids']
            for i in range(len(layers)):
                layer = layers[i].cuda()
                outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
                layers[i] = layer.cpu()
                inps = outs
                torch.cuda.empty_cache()
            _sync_if_cuda("cuda")
            total_forward_time += (time.perf_counter() - t0)
            total_tokens += int(batch.numel())
            total_target_tokens += int(batch[:, 1:].numel())
            hidden_states = norm(outs)
            lm_logits = lm_head(hidden_states)
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous().cuda()
                loss_sum = loss_fct(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                nll_sum += float(loss_sum.item())
                nll_count += int(shift_labels.numel())
            else:
                print("warning: nan or inf in lm_logits")
        if nll_count == 0:
            ppl = float("inf")
        else:
            ppl = float(np.exp(nll_sum / max(nll_count, 1)))
        ppls[dataset] = ppl
        tps_all = total_tokens / max(total_forward_time, 1e-12)
        tps_target = total_target_tokens / max(total_forward_time, 1e-12)
        throughput[dataset] = {
            "tokens_per_sec_all": tps_all,
            "tokens_per_sec_target": tps_target,
            "forward_time_sec": total_forward_time,
        }
        print(
            f"[{dataset}] throughput: "
            f"all_tokens/s={tps_all:.2f}, target_tokens/s={tps_target:.2f}, "
            f"forward_time={total_forward_time:.2f}s"
        )
    print("PPL after pruning: {}".format(ppls))
    print("Throughput (tokens/s): {}".format(throughput))
    weight_mib = _model_weight_memory_bytes(model) / (1024.0 * 1024.0)
    alloc_mib = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    reserved_mib = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
    print("Weight Memory (params+buffers): {:.2f} MiB".format(weight_mib))
    print("CUDA Allocated Memory: {:.2f} MiB".format(alloc_mib))
    print("CUDA Reserved Memory: {:.2f} MiB\n".format(reserved_mib))

@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda"):
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    weight_memory = torch.cuda.memory_allocated()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
            throughput += end_time - start_time
            print("time: {}".format(end_time - start_time))
    print("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
    print("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
    print("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
    print("Throughput: {} tokens/sec".format(token_num / throughput))
