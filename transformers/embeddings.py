import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

def load_glove_embeddings(path: str = 'glove.6B.300d.txt') -> tuple[Dict[str, int], torch.Tensor]:
    
    word2idx = {}
    embeddings_list = []
    
    word2idx['<pad>'] = 0
    embeddings_list.append(torch.zeros(300))
    
    word2idx['<unk>'] = 1
    embeddings_list.append(torch.randn(300))
    
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 2):
            values = line.split()
            word = values[0]
            vector = torch.FloatTensor([float(val) for val in values[1:]])
            
            word2idx[word] = i
            embeddings_list.append(vector)
            
    embeddings = torch.stack(embeddings_list)
    return word2idx, embeddings

class GloveEmbedding(nn.Module):
    
    def __init__(
        self, 
        word2idx: Dict[str, int],
        embeddings: torch.Tensor,
freeze: bool = True
    ):
        
        super().__init__()
        self.word2idx = word2idx
        self.embedding = nn.Embedding.from_pretrained(
            embeddings,
            padding_idx=0,
            freeze=freeze
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.embedding(x)
    
    def encode_text(self, text: str) -> torch.Tensor:
    
        words = text.lower().split()
        indices = [self.word2idx.get(word, 1) for word in words]
        return torch.tensor(indices)

