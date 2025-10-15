import torch
from torch import nn
from torch.nn.functional import layer_norm, softmax


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


class ViT(nn.Module):

    def __init__(
        self,
        img_size: int,
        chunk_size: int,
        num_heads: int,
        mlp_rank: int,
        num_classes: int,
    ):

        super().__init__()
        self.img_size = img_size
        self.chunk_size = chunk_size

        if img_size % chunk_size:
            raise Exception("`chunk_size` must divide `img_size`")

        if (chunk_size**2) % num_heads:
            raise Exception("`num_heads` must divide the square of `chunk_size`")

        num_patches = (img_size // chunk_size) ** 2
        self.d_model = chunk_size ** 2

        self.class_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, self.d_model))

        self.weights = nn.Parameter(torch.rand(num_heads, chunk_size ** 2, chunk_size ** 2))
        self.biases = nn.Parameter(torch.rand(num_heads, chunk_size ** 2))

        self.attention = MultiHeadAttention(num_heads, chunk_size**2)

        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.d_model, mlp_rank),
            nn.GELU(),
            nn.Linear(mlp_rank, self.d_model),
        )

        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)

        self.linear = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Extract patches
        x = self.crop(x)

        # Apply patch embedding (linear projection)
        x = torch.einsum("bni,hni->bhn", x, self.weights) + self.biases.unsqueeze(0)
        x = x.reshape(batch_size, -1, self.d_model)

        # Prepend class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.positional_embeddings

        # Transformer encoder block
        x = x + self.attention(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))
        x = x + self.encoder_mlp(self.layer_norm2(x))

        # Classification head (use only class token)
        cls_token_output = x[:, 0]
        x = self.linear(cls_token_output)

        return softmax(x, dim=-1)

    def crop(self, imgs):
        # imgs shape: (batch, channels, height, width)
        batch_size = imgs.size(0)
        num_chunks_per_side = self.img_size // self.chunk_size

        # Reshape to extract patches
        # (batch, channels, num_patches_h, chunk_size, num_patches_w, chunk_size)
        x = imgs.reshape(batch_size, -1, num_chunks_per_side, self.chunk_size,
                        num_chunks_per_side, self.chunk_size)

        # Rearrange to (batch, num_patches, patch_dim)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        num_patches = num_chunks_per_side ** 2
        x = x.reshape(batch_size, num_patches, -1)

        return x
