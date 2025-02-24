from torch import nn
import torch
import numpy as np 
from embeddings import load_glove_embeddings
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

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)


    def forward(self, x, y=None):

        if y is None:
            y = x
        
        Q = self.q(x)
        K = self.k(y)
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

        super.__init__()

        self.self_attention = MultiHeadAttention(
            num_heads=num_heads, 
            d_q=d_q, 
            d_k=d_q, 
            masked=True
        )
        self.n1 = LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            d_q=d_q,
            d_k=d_k,
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
        d_k: int
    ) -> None:

        super().__init__()

        self.word2idx, self.embeddings = load_glove_embeddings()
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.d_model = self.embeddings.size(1)

        self.input_embedding = nn.Embedding.from_pretrained(self.embeddings, freeze=True)
        self.output_embedding = self.input_embedding
        
        self.encoders = nn.ModuleList([TransformersEncoder(enc_heads, d_q, self.d_model) for _ in range(encoder_depth)])
        self.decoders = nn.ModuleList([TransformersDecoder(dec_heads, d_q, d_k, self.d_model) for _ in range(decoder_depth)])
        
        self.linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, src, tgt):
        src_embedded = self.input_embedding(src)
        tgt_embedded = self.output_embedding(tgt)
        
        src_pos = self.pos_encoding[:, :src_embedded.size(1)]
        tgt_pos = self.pos_encoding[:, :tgt_embedded.size(1)]
        
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

    def generate(self, src, max_length: int = 100):
        self.eval()
        with torch.no_grad():
            src_embedded = self.input_embedding(src)
            src_pos = self.pos_encoding[:, :src_embedded.size(1)]
            src_embedded = src_embedded + src_pos
            
            enc_output = src_embedded
            for encoder in self.encoders:
                enc_output = encoder(enc_output)
            
            tgt = torch.full((src.size(0), 1), self.word2idx['<start>'], device=src.device)
            
            for _ in range(max_length):
                tgt_embedded = self.output_embedding(tgt)
                tgt_pos = self.pos_encoding[:, :tgt_embedded.size(1)]
                tgt_embedded = tgt_embedded + tgt_pos
                
                dec_output = tgt_embedded
                for decoder in self.decoders:
                    dec_output = decoder(dec_output, enc_output)
                
                output = self.linear(dec_output)
                next_token = output[:, -1].argmax(dim=-1)
                
                if next_token.item() == self.word2idx['<end>']:
                    break
                    
                tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            
            return tgt
