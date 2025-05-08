import torch
from torch import nn
from embeddings import Embeddings
from helper import positional_encoding


class Attention(nn.Module):

    def __init__(self, 
            d_model: int,
            d_head: int,
            masked: bool = False
        ):

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
        scores = scores / (self.d_head ** 0.5)

        size = Q.size[1]

        if self.masked:
            mask = torch.triu(torch.ones(size, size), diagonal=1).bool().to(q.device)
            scores = scores.masked_fill(mask, float("-inf"))

        scores = torch.softmax(scores, dim=2)

        return torch.matmul(scores, V)


class MultiHeadAttention(nn.Module):

    def __init__(self, 
            num_heads: int, 
            d_model: int, 
            masked: bool = False
        ):

        super().__init__()
        
        self.num_heads = num_heads  
        self.d_model = d_model
        
        if d_model % num_heads:
            raise Exception("`d_model` must be divisible by `num_heads`!")
        
        self.d_head = d_model // num_heads
        
        self.heads = nn.ModuleList([Attention(self.d_model, self.d_head, masked) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):

        head_chunks = [head(q, k, v) for head in self.heads]
        concatenated = torch.cat(head_chunks, dim=-1)

        return self.proj(concatenated)


class LayerNorm(nn.Module):

    def __init__(self, d_model: int):

        super().__init__()
        
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
            d_model: int, 
        ):

        super().__init__()
        
        self.mh_attention = MultiHeadAttention(
            num_heads,
            d_model=d_model,
            masked=False
        )
        self.ln1 = LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = LayerNorm(d_model)
        

    def forward(self, x):
        
        x = x + self.mh_attention(x, x, x)
        x = self.ln1(x)

        x = x + self.ffn(x)
        x = self.ln2(x)

        return x


class TransformersDecoder(nn.Module):

    def __init__(self, 
            num_heads: int, 
            d_model: int
        ):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            num_heads, 
            d_model,
            masked=True
        )
        self.n1 = LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(
            num_heads,
            d_model,
            masked=False
        )
        self.n2 = LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.n3 = LayerNorm(d_model)

    def forward(self, x, y):

        x = x + self.self_attention(x, x, x)
        x = self.n1(x)
        
        x = x + self.cross_attention(x, y, y)
        x = self.n2(x)

        x = x + self.ffn(x)
        x = self.n3(x)

        return x


class Transformers(nn.Module):

    def __init__(self,
        encoder_depth: int,
        decoder_depth: int,
        enc_heads: int,
        dec_heads: int,
        d_model: int,
        vocab_size: int,
        tokens_path: str | None = None
    ) -> None:

        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        if tokens_path is not None:
            self.embeddings = Embeddings(
                d_model=d_model,
                vocab_size=vocab_size,
                saved_tokens_path=tokens_path
            )
        else:
            self.embeddings = None

        self.encoders = nn.ModuleList([TransformersEncoder(enc_heads, self.d_model) for _ in range(encoder_depth)])
        self.decoders = nn.ModuleList([TransformersDecoder(dec_heads, self.d_model) for _ in range(decoder_depth)])

    def forward(self, src, tgt):

        if self.embeddings is None:
            raise Exception("Embeddings must be created first")

        src_embedded = self.embeddings.encode(src)
        tgt_embedded = self.embeddings.encode(tgt)
        
        src_pos = positional_encoding(src_embedded.size(1), self.d_model).to(device=src.device)
        tgt_pos = positional_encoding(tgt_embedded.size(1), self.d_model).to(device=src.device)
        
        src_embedded = src_embedded + src_pos
        tgt_embedded = tgt_embedded + tgt_pos
        
        enc_output = src_embedded
        for encoder in self.encoders:
            enc_output = encoder(enc_output)
            
        dec_output = tgt_embedded
        for decoder in self.decoders:
            dec_output = decoder(dec_output, enc_output)
        
        return self.embeddings.decode(dec_output)

    def initiate_embeddings(self, embeddings):

        self.embeddings = embeddings

