import torch
from torch import nn 


class Embeddings(nn.Module):

    def __init__(
            self,
            d_model: int,         # unchangable
            vocab_size: int       # can be changed with an internal funcion
        ):

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size


    def forward(self, x):

        pass
    
    def extend_vocabulary(self, new_tokens):

        pass

    def reduce_vocabulary(self, del_tokens):

        pass
