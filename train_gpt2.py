from typing import Optional
from dataclasses import dataclass
import math
import inspect

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
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

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
        return logits

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
        print(
            f"num weight-decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-weight-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters"
        )
        # create AdamW optimizer and use the fused version (does not loop over the parameters) if it is avaialable
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )  # betas and eps taken from openai paper
        return optimizer


class DataLoaderLite:
    def __init__(self, B: int, T: int):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # loader state
        self.current_position = 0

    def next_batch(
        self,
    ):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # update state
        self.current_position += B * T
        # reset state if next batch will be out of bounds
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y


if __name__ == "__main__":
    import time

    # Set device
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

    # data loader
    train_loader = DataLoaderLite(B=4, T=1024)

    # try TF32
    if torch.cuda.is_tf32_supported():
        print("TF32 is supported")
        torch.set_float32_matmul_precision("high")

    # model init
    model = GPT(
        GPTConfig(vocab_size=50304)
    )  # 50304 = 128 * 393 = 2^7 * 393 (a nice number, better for cuda calculations)
    model.to(device)
    # use compiled model
    model.compile()

    # learning rate schedule from openai
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50

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
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=max_lr, device=device
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # optimize
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # use BF16
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        # clip gradients at 1.0 (same as openai)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scheduler.step()
        optimizer.step()
        # wait for the gpu to synchronize
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # time diff in ms
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(
            f"step {i:4d} | loss: {loss.item():.6f} | lr: {scheduler.get_last_lr()[0]:.4e} | norm: {norm.item():.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )

    import sys

    sys.exit(0)

    num_return_sequences = 5
    max_length = 30

    # model = GPT.from_pretrained("gpt2")

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    # generate!
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)  # (B, T, vocab_size)
            # We are only interested in the logits at the last token
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # top-k sampling (k = 50)
            topk_probs, topk_indices = torch.topk(
                probs, k=50, dim=-1
            )  # (B, 50), (B, 50)
            next_token = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
            next_token = torch.gather(topk_indices, 1, next_token)  # (B, 1)
            # append next token to input
            x = torch.cat((x, next_token), dim=1)  # (B, T+1)

    # print the generated text
    for i in range(num_return_sequences):
        generated_tokens = x[i].tolist()
        generated_text = enc.decode(generated_tokens)
        print(">", generated_text)
