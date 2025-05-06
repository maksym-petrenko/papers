from torch import nn
import torch
from embeddings import Embeddings
from helper import positional_encoding


class Attention(nn.Module):

    def __init__(self, 
            d_q: int,
            d_k: int, 
            d_model: int, 
            masked: bool = False
        ):

        super().__init__()
        
        self.d_q = d_q
        self.d_k = d_k
        self.d_model = d_model

        if masked and d_q != d_k:
            raise Exception("Only self attention supports masking")
        self.masked = masked

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)


    def forward(self, x, y=None):

        if y is None:
            y = x
        
        Q = self.q(x)
        K = self.k(y)
        V = self.v(x)
        
        K = K.transpose(-2, -1)

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
            d_q: int, 
            d_k: int,
            d_model: int, 
            masked: bool = False
        ):

        super().__init__()
        
        self.num_heads = num_heads  
        self.d_k = d_k
        self.d_q = d_q
        self.d_model = d_model
        
        if d_model % num_heads:
            raise Exception("`d_model` must be divisible by `num_heads`!")

        self.d_heads = d_model / num_heads
        
        self.heads = nn.ModuleList([Attention(d_q, d_k, self.d_heads, masked) for _ in range(num_heads)])
        self.WO = nn.Linear(d_model, d_model)

    def forward(self, x, y=None):

        if y is None:
            x = y

        chunks = torch.split(x, self.d_heads, dim=-1)
        head_chunks = [head(chunk, y) for head, chunk in zip(self.num_heads, chunks)]
        
        concatenated = torch.cat(head_chunks, dim=-1)

        x = self.WO(concatenated)

        return x


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
            d_q: int,
            d_model: int, 
        ):

        super().__init__()
        
        self.mh_attention = MultiHeadAttention(
            num_heads=num_heads,
            d_q=d_q,
            d_k=d_q,
            d_model=d_model,
            masked=False
        )
        self.ln1 = LayerNorm(d_model)

        self.ffn = nn.Linear(d_model, d_model)
        self.ln2 = LayerNorm(d_model)
        

    def forward(self, x):
        
        x = x + self.mh_attention(x)
        x = self.ln1(x)

        x = x + self.ffn(x)
        x = self.ln2(x)

        return x


class TransformersDecoder(nn.Module):

    def __init__(self, num_heads: int, d_q: int, d_k: int, d_model):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            num_heads=num_heads, 
            d_q=d_q, 
            d_k=d_q,
            d_model=d_model,
            masked=True
        )
        self.n1 = LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            d_q=d_q,
            d_k=d_k,
            d_model=d_model,
            masked=False
        )
        self.n2 = LayerNorm(d_model)
        
        self.ffn = nn.Linear(d_model, d_model)
        self.n3 = LayerNorm(d_model)

    def forward(self, x, y):

        x = x + self.self_attention(x)
        x = self.n1(x)
        
        x = x + self.cross_attention(x, y)
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
        d_q: int,
        d_k: int,
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

        self.encoders = nn.ModuleList([TransformersEncoder(enc_heads, d_q, self.d_model) for _ in range(encoder_depth)])
        self.decoders = nn.ModuleList([TransformersDecoder(dec_heads, d_q, d_k, self.d_model) for _ in range(decoder_depth)])
        
        self.linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, src, tgt):
        src_embedded = self.embeddings.encode(src)
        tgt_embedded = self.embeddings.encode(tgt)
        
        src_pos = positional_encoding[:, :src_embedded.size(1)]
        tgt_pos = positional_encoding[:, :tgt_embedded.size(1)]
        
        src_embedded = src_embedded + src_pos
        tgt_embedded = tgt_embedded + tgt_pos
        
        enc_output = src_embedded
        for encoder in self.encoders:
            enc_output = encoder(enc_output)
            
        dec_output = tgt_embedded
        for decoder in self.decoders:
            dec_output = decoder(dec_output, enc_output)
            
        output = self.linear(dec_output)
        
        return output

