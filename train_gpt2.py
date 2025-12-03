from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))  # (B, nh, T, T)
        att = att.masked_fill(mask=self.bias[:, :, :T, :T] == 0, value=float("-inf"))  # type: ignore
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
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

    def forward(self, idx: torch.Tensor):
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
        return logits

    @classmethod
    def from_pretrained(cls, model_type) -> GPT:
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


if __name__ == "__main__":
    num_return_sequences = 5
    max_length = 30

    model = GPT.from_pretrained("gpt2")
    model.eval()
    model.to("cuda")

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to("cuda")

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
