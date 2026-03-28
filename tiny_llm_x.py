import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Low-rank embedding
class LowRankEmbedding(nn.Module):
    """Factorized embedding: A (vocab, rank) @ B (rank, dim)."""
    def __init__(self, vocab_size: int, dim: int, rank: int, tie: bool = True):
        super().__init__()
        self.A = nn.Parameter(torch.empty(vocab_size, rank))
        self.B = nn.Parameter(torch.empty(rank, dim))
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.normal_(self.B, mean=0.0, std=0.02)
        self.tie = tie
        if tie:
            self.register_parameter("weight", None)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (B, S)
        emb = torch.nn.functional.embedding(token_ids, self.A)  # (B, S, rank)
        return torch.matmul(emb, self.B)  # (B, S, dim)

    @property
    def output_weight(self):
        # weight for logits: (vocab, dim)
        return torch.matmul(self.A, self.B)

# SwiGLU MLP
class SwiGLU(nn.Module):
    """Efficient SwiGLU MLP: (xW1)*(sigmoid(xW2)) @ W3"""
    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = dim * hidden_mult
        self.W1 = CastedLinear(dim, hidden, bias=False)
        self.W2 = CastedLinear(dim, hidden, bias=False)
        self.W3 = CastedLinear(hidden, dim, bias=False)
        self.W3._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        h1 = self.W1(x)
        h2 = self.W2(x)
        gated = h1 * torch.sigmoid(h2)
        return self.W3(gated)

# Shared transformer block (recurrent depth)
class SharedTransformerBlock(nn.Module):
    """Single transformer block reused across steps."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = SwiGLU(dim)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

# TinyLLM-X model
class TinyLLMXModel(nn.Module):
    """TinyLLM‑X model with shared block and low‑rank embedding."""
    def __init__(self, vocab_size: int, dim: int, num_heads: int, num_kv_heads: int, rank: int,
                 num_steps: int = 12, tie_embeddings: bool = True, logit_softcap: float = 30.0,
                 rope_base: float = 10000.0, qk_gain_init: float = 1.5):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError("logit_softcap must be positive")
        self.dim = dim
        self.num_steps = num_steps
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.emb = LowRankEmbedding(vocab_size, dim, rank, tie=tie_embeddings)
        self.step_emb = nn.Parameter(torch.zeros(num_steps, dim))
        self.shared_block = SharedTransformerBlock(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.final_norm = RMSNorm()
        if not tie_embeddings:
            self.lm_head = CastedLinear(dim, vocab_size, bias=False)
            self.lm_head._zero_init = True
        else:
            self.lm_head = None
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.step_emb, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.emb(input_ids)  # (B, S, dim)
        x0 = x.clone()
        for step in range(self.num_steps):
            step_vec = self.step_emb[step].view(1, 1, -1)
            x = x + step_vec
            x = self.shared_block(x, x0)
        x = self.final_norm(x).reshape(-1, self.dim)
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits = F.linear(x, self.emb.output_weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
