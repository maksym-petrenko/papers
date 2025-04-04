from torch import nn 
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from helper import is_english_or_french
import heapq


class Embeddings(nn.Module):

    def __init__(
            self,
            d_model: int,                                     # unchangable
            vocab_size: int,                                  # can be changed with an internal funcion
            dataset: str,                                     # path to the dataset, makes sense only if vocab_size is defined
            min_token_occurrence: float = 1e-4,               # min number of average occurances of the token in the dataset per line
            verbose: int = 1                                  # value of 0 or 1 representing whether info is printed    
        ) -> None:

        if verbose not in [0, 1]:
            raise Exception("Verbose must be equal to `0` or `1`!")

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_to_id = None
        self.embeddings = None
        self.projection = nn.Linear(d_model, vocab_size)
        

    def encode(self, text: str):

        pass
    
    def decode(self, vect) -> str:

        pass


    def extend_vocabulary(self, new_tokens):

        pass

    def reduce_vocabulary(self, del_tokens):

        pass
