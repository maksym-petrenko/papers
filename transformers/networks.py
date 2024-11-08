from torch import nn
import torch
import numpy as np 


class Attention(nn.Module):

    def __init__(self, input_dim: int, embeddings_size: int):

        super().__init__()
        
        self.input_dim = input_dim
        self.embeddings_size = embeddings_size

        self.q = nn.Linear(embeddings_size, embeddings_size)
        self.k = nn.Linear(embeddings_size, embeddings_size)
        self.v = nn.Linear(embeddings_size, embeddings_size)


    def forward(self, x, masked: bool = False):
        
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        K = torch.transpose(K, -2, -1)

        scores = torch.matmul(Q, K)
        scores = scores / torch.sqrt(self.embeddings_size)
        
        if masked:
            mask = torch.tril(self.input_dim, self.input_dim).bool().to(x.device)
            scores = masked_fill(mask, float("-inf"))

        scores = nn.Softmax(scores, dim=2)

        x = torch.matmul(scores, V)
        return x


