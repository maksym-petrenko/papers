from torch import nn
import torch
import numpy as np 


class Attention(nn.Module):

    def __init__(self, input_dim: int, embeddings_size: int, masked: bool = False):

        super().__init__()
        
        self.input_dim = input_dim
        self.embeddings_size = embeddings_size
        self.masked = masked

        self.q = nn.Linear(embeddings_size, embeddings_size)
        self.k = nn.Linear(embeddings_size, embeddings_size)
        self.v = nn.Linear(embeddings_size, embeddings_size)


    def forward(self, x):
        
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        K = torch.transpose(K, -2, -1)

        scores = torch.matmul(Q, K)
        scores = scores / torch.sqrt(self.embeddings_size)
        
        if self.masked:
            mask = torch.tril(self.input_dim, self.input_dim).bool().to(x.device)
            scores = masked_fill(mask, float("-inf"))

        scores = nn.Softmax(scores, dim=2)

        x = torch.matmul(scores, V)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, input_dim: int, embeddings_size: int, masked: bool = False):

        super().__init__()
        
        self.num_heads = num_heads  
        self.input_dim = input_dim
        self.embeddings_size = embeddings_size
        
        if embeddings_size % heads:
            raise Ecxeption("`embeddings_size` must be divisible by `heads`!")

        self.local_embed_size = embeddings_size / num_heads
        
        self.heads = nn.ModuleList([Attention(input_dim, self.local_embed_size, masked) for _ in range(num_heads)])
        self.WO = nn.Linear(embeddings_size, embeddings_size)

    def forward(self, x):

        head_chunks = torch.split(x, self.local_embed_size, dim=-1)
        head_chunks = [head(chunk) for head, chunk in zip(self.num_heads, head_chunks)]
        
        concatenated = torch.cat(head_chunks, dim=-1)

        x = self.WO(concatenated)

        return x

