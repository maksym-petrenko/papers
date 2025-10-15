import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self, d_model: int, d_head: int, masked: bool = False):

        super().__init__()

        if not (d_model / d_head).is_integer():
            raise Exception("`d_model` must be divisible by `d_heads`!")

        self.d_model = d_model
        self.d_head = d_head
        self.masked = masked

        self.q = nn.Linear(d_model, d_head, bias=False)
        self.k = nn.Linear(d_model, d_head, bias=False)
        self.v = nn.Linear(d_model, d_head, bias=False)

    def forward(self, q, k, v):

        Q = self.q(q)
        K = self.k(k)
        V = self.v(v)

        K = K.transpose(-2, -1)

        scores = torch.matmul(Q, K)
        scores = scores / (self.d_head**0.5)

        size = Q.size()[-2]

        if self.masked:
            mask = torch.triu(torch.ones(size, size), diagonal=1).bool().to(q.device)

            if len(scores.size()) == 3:
                mask = mask.repeat(scores.size(0), 1, 1)

            scores = scores.masked_fill(mask, float("-inf"))

        scores = torch.softmax(scores, dim=-1)

        return torch.matmul(scores, V)


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, d_model: int, masked: bool = False):

        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        if d_model % num_heads:
            raise Exception("`d_model` must be divisible by `num_heads`!")

        self.d_head = d_model // num_heads

        self.heads = nn.ModuleList(
            [Attention(self.d_model, self.d_head, masked) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):

        head_chunks = [head(q, k, v) for head in self.heads]
        concatenated = torch.cat(head_chunks, dim=-1)

        return self.proj(concatenated)
