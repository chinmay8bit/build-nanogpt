from typing import Optional
from dataclasses import dataclass
import math
import inspect
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length (contex window)
    vocab_size: int = (
        50257  # 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), f"Embedding dimension {config.n_embd} is not divisible by the number of heads {config.n_head}"
        # key, query, value projections (3) for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not bias, but the causal mask (obviously not a trainable param, hence put in buffer)
        # not needed anymore as we are now using pytorch's flash attention method
        # self.register_buffer(
        #     "bias",
        #     torch.tril(torch.ones(config.block_size, config.block_size)).view(
        #         1, 1, config.block_size, config.block_size
        #     ),
        # )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # nh = number of heads (n_head)
        # hs = head size, size of projections of a single head (n_embed // n_head)
        head_size = C // self.n_head
        # Calculate the query, key, value projections
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = torch.split(qkv, split_size_or_sections=C, dim=2)  # (B, T, C)
        # batch the q, k, v by heads
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))  # (B, nh, T, T)
        # att = att.masked_fill(mask=self.bias[:, :, :T, :T] == 0, value=float("-inf"))  # type: ignore
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # Replace with flash attention!
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # reassemble the output of all heads into the original shape
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        # output projection
        y = self.c_proj(y)  # (B, T, C)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
                wpe=nn.Embedding(
                    config.block_size, config.n_embd
                ),  # Position embeddings
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight  # type: ignore # (pointer copy) they both point to the same tensor now

        # init params
        # self.apply(self._init_weights)
        # I use this instead of self.apply because I need access to module name as well
        for name, module in self.named_modules():
            self._init_weights(name, module)

    def _init_weights(self, name: str, module):
        # mirror openai gpt2 initialization
        if isinstance(module, nn.Linear):
            std = 0.02
            if name.endswith(
                ".c_proj"
            ):  # These are the final projections in the attn and mlp blocks followed by the residual connections
                std *= 1.0 / math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):  # this is the positional embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Length of sequence {T} exceeds maximum context window of {self.config.block_size}"
        # calculate the token and position embeddings
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)  # type: ignore # token embeddings of shape (B, T, C)
        pos_emb = self.transformer.wpe(pos)  # type: ignore # position embeddings of shape (T, C)
        x = tok_emb + pos_emb
        # forward the transformer blocks
        for block in self.transformer.h:  # type: ignore
            x = block(x)
        # Apply the final layernorm and lm head
        x = self.transformer.ln_f(x)  # type: ignore
        logits = self.lm_head(x)  # type: ignore # (B, T, vocab_size)
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, self.config.vocab_size),
                targets.view(B * T),
            )
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type) -> "GPT":
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        config = {
            "gpt2": GPTConfig(n_layer=12, n_head=12, n_embd=768),  # 124 params
            "gpt2-medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024),  # 350 params
            "gpt2-large": GPTConfig(n_layer=36, n_head=20, n_embd=1280),  # 774 params
            "gpt2-xl": GPTConfig(n_layer=48, n_head=25, n_embd=1600),  # 1558 params
        }[model_type]

        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf_model.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self, weight_decay, learning_rate, device
    ) -> torch.optim.Optimizer:
        """
        This adds weight decay to multi-dimensional params
        """
        # Pull out all trainable params
        params_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Separate out all parameters to those that will and won't experience regularizing weight decay
        decay_params = [p for p in params_dict.values() if p.dim() > 1]
        no_decay_params = [p for p in params_dict.values() if p.dim() <= 1]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        if master_process:
            print(
                f"num weight-decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-weight-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters"
            )
        # create AdamW optimizer and use the fused version (does not loop over the parameters) if it is avaialable
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )  # betas and eps taken from openai paper
        return optimizer


def load_tokens(filename):
    tokens_np = np.load(filename)
    tokens = torch.from_numpy(tokens_np).long()
    return tokens


# TODO: Add shuffing of documents and/or shards (relevant for multi-epoch runs)
class DataLoaderLite:
    def __init__(
        self,
        B: int,
        T: int,
        process_rank: int = 0,
        num_processes: int = 1,
        split: str = "train",
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames based on the split
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        self.shards = [os.path.join(data_root, s) for s in sorted(shards)]
        assert len(self.shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        # init
        self.reset()

    def reset(
        self,
    ):
        """
        Resets the dataloader to the initial state
        """
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(
        self,
    ):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # advance positions in the self.tokens tensor (tokens of current shard)
        self.current_position += B * T * self.num_processes
        # advance to next shard if next batch will be out of bounds
        if self.current_position + (B * T * self.num_processes) + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------------------
# simple launch:
# poetry run python train_gpt2.py
# DDP launch for e.g. 2 GPUs:
# poetry run torchrun --standalone --nproc_per_node=2 train_gpt2.py
# Note: might need to disable NCCL P2P if your provider has blocked direct GPU<->GPU P2P connections.
# This will result in some time penalty during gradient sync, but should get the code to work.
# NCCL_P2P_DISABLE=1 poetry run torchrun --standalone --nproc_per_node=2 train_gpt2.py
# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import time
    import argparse
    import uuid

    import tiktoken
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    import hellaswag

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint to load.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "-B",
        "--batch-size",
        type=int,
        default=16,
        help="Micro batch size per GPU.",
    )
    args = parser.parse_args()

    # Set device
    # set up DDP (distributed data parallel)
    # torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        # Set NCCL debug envs for more logs
        # os.environ["NCCL_DEBUG"] = "INFO"
        # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # the master process will be responsible for logging, checkpointing, etc.
        if master_process:
            print("Using DDP")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # define stuff for gradient accumulation
    total_batch_size_in_tokens = (
        524288  # 524288 = 2^19, ~0.5M (used by openai for 124M gpt2)
    )
    B = (
        args.batch_size
    )  # micro batch size (set this based on your available GPU memory)
    T = 1024  # sequence length
    assert (
        total_batch_size_in_tokens % (B * T * ddp_world_size) == 0
    ), "total batch size in tokens must be divisible by BxTxW (W = ddp world size)"
    grad_accum_steps = total_batch_size_in_tokens // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size (in tokens): {total_batch_size_in_tokens}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # data loader
    train_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
    )
    val_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
    )

    # try TF32
    if torch.cuda.is_tf32_supported():
        if master_process:
            print("TF32 is supported")
        torch.set_float32_matmul_precision("high")

    # model init
    model = GPT(
        GPTConfig(vocab_size=50304)
    )  # 50304 = 128 * 393 = 2^7 * 393 (a nice number, better for cuda calculations)
    model.to(device)
    # use compiled model
    model.compile()

    # load model checkpoint if provided
    if args.ckpt:
        if master_process:
            print(f"loading model checkpoint from {args.ckpt}...")
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])

    # wrap model in DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model: GPT = model.module if ddp else model  # type: ignore

    # learning rate schedule from openai
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = (
        715  # openai gpt2 warms up for the first 375M tokens. 375M / 524288 = ~715.
    )
    max_steps = (
        19073  # This is the steps for training for 10B tokens. 10B / 524288 = ~19073. (1 epoch)
    ) * args.epochs

    def lr_lambda(step):
        # 1) linear warmup for warmup steps
        if step < warmup_steps:
            return step / warmup_steps
        # 2) after cosine decay (max steps), continue at min_lr learning rate
        if step > max_steps:
            return min_lr / max_lr
        # 3) cosine decay after warmup steps (continues till max steps)
        decay_ratio = (step - warmup_steps) / (
            max_steps - warmup_steps
        )  # decay_ratio goes from 0 to 1
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff goes from 1 to 0
        lr = min_lr + coeff * (max_lr - min_lr)
        return lr / max_lr

    # create optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=max_lr, device=device
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # restore optimizer and scheduler state from model checkpoint if provided
    if args.ckpt:
        if master_process:
            print(f"loading optimizer and scheduler state from {args.ckpt}...")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # type: ignore
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # type: ignore

    # create gpt2 tokenizer (used for generation)
    enc = tiktoken.get_encoding("gpt2")

    # create the log directory for saving checkpoints and logs
    log_dir = os.path.join("logs", f"run_{uuid.uuid4().hex[:6]}")
    log_file = os.path.join(log_dir, f"log.txt")
    if master_process:
        os.makedirs(log_dir, exist_ok=True)
        print(f"logging to file: {log_file}")
        with open(log_file, "w") as f:  # this will clear the file
            pass

    # Set start step if resuming from checkpoint
    if args.ckpt:
        start_step = checkpoint["step"]  # type: ignore
        # Update train dataloader state to this start_step
        for step in range(start_step):
            for micro_step in range(grad_accum_steps):
                train_loader.next_batch()
    else:
        start_step = 0

    # optimize
    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        # evaluate validation loss every 250 steps
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = torch.tensor(0.0, device=device)
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(
                    f"step {step:4d}/{max_steps:4d} | validation loss: {val_loss_accum.item():.6f}"
                )
                with open(log_file, "a") as f:
                    f.write(
                        f"step {step:4d}/{max_steps:4d} | validation loss: {val_loss_accum.item():.6f}\n"
                    )
                # save model checkpoints every 5000 steps
                if (step != 0 and step % 5000 == 0) or last_step:
                    ckpt_path = os.path.join(
                        log_dir, "checkpoints", f"model_{step:05d}.pt"
                    )
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "config": raw_model.config,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                    }
                    print(f"saving model checkpoint to {ckpt_path}...")
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    torch.save(checkpoint, ckpt_path)

        # evaluate hellaswag every 250 steps
        if step % 250 == 0 or last_step:
            model.eval()
            num_correct_norm = torch.tensor(0, dtype=torch.int32, device=device)
            num_total = torch.tensor(0, dtype=torch.int32, device=device)
            for i, example in enumerate(hellaswag.iterate_examples("val")):
                # distribute examples across multiple GPUs (if available)
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render example
                _, tokens, mask, label = hellaswag.render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, _ = model(tokens)
                    pred, pred_norm, _, _ = hellaswag.get_most_likely_ending(
                        tokens, mask, logits
                    )
                num_total += 1
                if pred_norm == label:
                    num_correct_norm += 1
            # reduce the stats across all GPU processes
            if ddp:
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(
                    f"step {step:4d}/{max_steps:4d} | HellaSwag accuracy: {num_correct_norm.item()}/{num_total.item()}={acc_norm.item():.4f}"
                )
                with open(log_file, "a") as f:
                    f.write(
                        f"step {step:4d}/{max_steps:4d} | HellaSwag accuracy: {acc_norm.item():.4f}\n"
                    )

        # generate from the model every 250 steps
        if (step != 0 and step % 250 == 0) or last_step:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            # generate!
            sample_rng = torch.Generator(
                device=device
            )  # creating a custom rng generator for sampling, so that training rng state (global rng state) is unaffected
            sample_rng.manual_seed(42 + ddp_rank)  # unique seed for each gpu
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, _ = model(xgen)  # (B, T, vocab_size)
                    # We are only interested in the logits at the last token
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    probs = F.softmax(logits, dim=-1)
                    # top-k sampling (k = 50, huggingface gpt2 pipeline default setting)
                    topk_probs, topk_indices = torch.topk(
                        probs, k=50, dim=-1
                    )  # (B, 50), (B, 50)
                    next_token_idx_in_topk = torch.multinomial(
                        topk_probs, num_samples=1, generator=sample_rng
                    )  # (B, 1)
                    next_token = torch.gather(
                        topk_indices, 1, next_token_idx_in_topk
                    )  # (B, 1)
                    # append next token to input
                    xgen = torch.cat((xgen, next_token), dim=1)  # (B, T+1)
            # print the generated text
            for i in range(num_return_sequences):
                generated_tokens = xgen[i].tolist()
                generated_text = enc.decode(generated_tokens)
                print(f"rank {ddp_rank} sample {i}: {generated_text}")

        # training loop (one step of the optimization)
        model.train()
        optimizer.zero_grad()
        loss_accum = torch.tensor(0.0, device=device)
        for micro_step in range(grad_accum_steps):  # accumulate gradients
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # ERRATA (karpathy): Confusingly, model.require_backward_grad_sync is actually used by both the forward and backward pass.
            # Moved up the line so that it also gets applied to the forward pass.
            if ddp:
                # only synchronize at the last micro step per gpu
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1  # type: ignore
            # use BF16
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = (
                loss / grad_accum_steps
            )  # Scale down the loss since gradient accumulation results in gradient of loss sum (we want gradient of loss mean)
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            # calculate global average of loss accum (accross gpus)
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # clip gradients at 1.0 (same as openai)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.step()
        optimizer.step()
        # wait for the gpu to synchronize
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # time diff in ms
        tokens_processed = (
            train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        )
        tokens_per_sec = tokens_processed / (t1 - t0)
        if master_process:
            print(
                f"step {step+1:4d}/{max_steps:4d} | loss: {loss_accum.item():.6f} | lr: {scheduler.get_last_lr()[0]:.4e} | norm: {norm.item():.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a") as f:
                f.write(
                    f"step {step+1:4d}/{max_steps:4d} | train loss: {loss_accum.item():.6f}\n"
                )

    if ddp:
        dist.destroy_process_group()
