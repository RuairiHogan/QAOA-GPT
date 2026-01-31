import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class QAOAConfig():
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

    # FEATHER conditioning
    graph_dim: int = 0          # set >0 to enable graph conditioning

    # EOS / masking
    eos_token_id: int = -1      # set to >=0 to enable loss masking after EOS


# -------------------------
# Building blocks (nanoGPT-style)
# -------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, 1e-5
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: QAOAConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: QAOAConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = nn.functional.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: QAOAConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -------------------------
# QAOA GPT (new class name)
# -------------------------
class QAOAgpt(nn.Module):
    """
    Drop-in GPT-like model with:
      - optional graph embedding conditioning (FEATHER)
      - generate() stops on eos_token_id
      - forward() can mask loss after EOS (optional)
    """
    def __init__(self, config: QAOAConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Graph embedding projection (FEATHER -> n_embd)
        self.graph_proj = None
        if config.graph_dim and config.graph_dim > 0:
            self.graph_proj = nn.Linear(config.graph_dim, config.n_embd, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_loss_mask_after_eos(self, targets: torch.Tensor, eos_token_id: int) -> torch.Tensor:
        """
        Creates a mask of shape (B, T) that is 1 up to and including the first EOS,
        and 0 strictly after it. If EOS does not appear, mask is all 1s.
        """
        # targets: (B, T)
        B, T = targets.shape
        eos = (targets == eos_token_id)  # (B, T) boolean

        # Find first EOS index per row; if none, set to T (meaning keep all tokens)
        # We do this by taking argmax over a modified tensor.
        # Trick: convert eos to int and use cumulative sums.
        eos_int = eos.int()
        csum = torch.cumsum(eos_int, dim=1)          # increases after first EOS
        after_first = (csum >= 1)                    # True at and after first EOS
        # keep tokens up to and including EOS => mask = 1 where after_first is True,
        # BUT we want to zero out *strictly after* EOS, so shift.
        # We'll compute "past_eos" = (csum >= 2) OR (csum >=1 and position > first_eos)
        # Easier: build mask where csum <= 1 (means before first EOS or at first EOS)
        mask = (csum <= 1).float()                   # 1 up to first EOS inclusive, 0 afterwards
        return mask  # (B, T)

    def forward(self, idx, graph_emb=None, targets=None, eos_token_id=None):
        """
        idx: (B, T) token ids
        graph_emb: (B, graph_dim) float tensor, required if graph_dim>0
        targets: (B, T) next-token ids
        eos_token_id: optional override; if provided (or config.eos_token_id>=0), mask loss after EOS
        """
        device = idx.device
        B, T = idx.size()

        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} > block size {self.config.block_size}")

        # Enforce graph embedding alignment at runtime
        if self.graph_proj is not None:
            if graph_emb is None:
                raise ValueError("graph_emb must be provided when graph_dim > 0")
            if graph_emb.shape[0] != B:
                raise ValueError(f"graph_emb batch mismatch: graph_emb has {graph_emb.shape[0]}, idx has {B}")

        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        tok_emb = self.transformer.wte(idx)            # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)            # (T, n_embd) -> broadcast to (B, T, n_embd)
        x = tok_emb + pos_emb

        # Inject FEATHER graph embedding
        if self.graph_proj is not None:
            g = self.graph_proj(graph_emb)             # (B, n_embd)
            g = g.unsqueeze(1).expand(-1, T, -1)       # (B, T, n_embd)
            x = x + g

        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)                   # (B, T, n_embd)
        logits = self.lm_head(x)                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Decide EOS id for masking
            mask_eos_id = eos_token_id
            if mask_eos_id is None:
                mask_eos_id = self.config.eos_token_id

            if mask_eos_id is not None and mask_eos_id >= 0:
                # Compute per-token loss then mask out tokens after EOS
                # loss_per_token: (B, T)
                loss_per_token = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="none"
                ).view(B, T)

                mask = self._make_loss_mask_after_eos(targets, mask_eos_id)  # (B, T)
                # Avoid division by zero if mask is all zeros (shouldn't happen unless sequence starts after EOS)
                denom = mask.sum().clamp(min=1.0)
                loss = (loss_per_token * mask).sum() / denom
            else:
                # Standard GPT loss over all tokens
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        graph_emb=None,
        temperature=1.0,
        top_k=None,
        eos_token_id=None,
    ):
        """
        Autoregressive generation. Stops early if eos_token_id is produced.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond, graph_emb=graph_emb)

            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None:
                # stop if ALL sequences in batch produced EOS this step
                if (idx_next == eos_token_id).all():
                    break

        return idx