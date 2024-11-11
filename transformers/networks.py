from torch import nn
import torch
import numpy as np 


class Attention(nn.Module):

    def __init__(self, 
            q_dim: int,
            k_dim: int, 
            d_model: int, 
            masked: bool = False
        ):

        super().__init__()
        
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.d_model = d_model

        if masked and q_dim != k_dim:
            raise Exception("Only self attention supports masking")
        self.masked = masked

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)


    def forward(self, x):
        
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        K = torch.transpose(K, -2, -1)

        scores = torch.matmul(Q, K)
        scores = scores / torch.sqrt(self.d_model)
        
        if self.masked:
            mask = torch.tril(self.k, self.k).bool().to(x.device)
            scores = masked_fill(mask, float("-inf"))

        scores = nn.Softmax(scores, dim=2)

        x = torch.matmul(scores, V)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, 
            num_heads: int, 
            q_dim: int, 
            k_dim: int,
            d_model: int, 
            masked: bool = False
        ):

        super().__init__()
        
        self.num_heads = num_heads  
        self.k_dim = k_dim
        self.q_dim = q_dim
        self.d_model = d_model
        
        if d_model % heads:
            raise Ecxeption("`d_model` must be divisible by `heads`!")

        self.d_heads = d_model / num_heads
        
        self.heads = nn.ModuleList([Attention(q_dim, k_dim, self.d_heads, masked) for _ in range(num_heads)])
        self.WO = nn.Linear(d_model, d_model)

    def forward(self, x):

        head_chunks = torch.split(x, self.d_heads, dim=-1)
        head_chunks = [head(chunk) for head, chunk in zip(self.num_heads, head_chunks)]
        
        concatenated = torch.cat(head_chunks, dim=-1)

        x = self.WO(concatenated)

        return x


class LayerNorm(nn.Module):

    def __init__(self, d_model: int):

        super.__init__()
        
        self.g = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):

        mean = torch.mean(x, -1, keepdim=True)
        var = torch.var(x, -1, keepdim=True)
        
        epsilon = 1e-12

        x = (x - mean) * self.g / torch.sqrt(var + epsilon) + self.b 
        
        return x


class TransformersEncoder(nn.Module):

    def __init__(self, 
            num_heads: int, 
            input_dim: int, 
            d_model: int, 
        ):

        super().__init__()
        
        self.mh_attention = MultiHeadAttention(
            num_heads=num_heads,
            input_dim=input_dim,
            d_model=d_model,
            masked=False
        )
        self.ln1 = LayerNorm(d_model)

        self.ffn = nn.Linear(d_model, d_model)
        self.ln2 = LayerNorm(d_model)
        

    def forward(self, x):
        
        y = self.mh_attention(x)
        x = x + y
        x = self.ln1(x)

        y = self.ffn(x)
        x = x + y
        x = self.ln2(x)

        return x


class TransformersDecoder(nn.Module):

    def __init__(self):

        super.__init__()

    def forward(self, x): 

        pass
