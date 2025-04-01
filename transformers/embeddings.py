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
            min_token_occurrence: float = 1e-4                 # min number of average occurances of the token in the dataset per line
            ) -> None:

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_to_id = None
        self.embeddings = None
        self.projection = nn.Linear(d_model, vocab_size)

        df = pd.read_csv(dataset)
        df = df.iloc[:1000000]
        min_samples = int(min_token_occurrence * len(df))
        data = defaultdict(int)

        for _, line in tqdm(df.iterrows(), total=len(df)):
            for char in (str(line["en"]) + str(line["fr"])):
                data[char] += 1
        data = {char: n for char, n in data.items() if n >= min_samples}
        
        tokens = list(data.keys())
        tokens = [sorted([token for token in tokens if is_english_or_french(token)])]
        max_token_length = 1

        while sum([len(i) for i in tokens]) < vocab_size - 1:
            max_token_length += 1
            data = defaultdict(int)

            for _, line in tqdm(df.iterrows(), total=len(df)):
                for i in range(len(str(line["en"])) - max_token_length):
                    data[str(line["en"])[i:i + max_token_length]] += 1
                for i in range(len(str(line["fr"])) - max_token_length):
                    data[str(line["fr"])[i:i + max_token_length]] += 1
                
            data = {token: n for token, n in data.items() if n >= min_samples}

            if len(data) + sum([len(i) for i in tokens]) > vocab_size - 1:
                n = vocab_size - 1 - sum([len(i) for i in tokens]) 
                data = [k for k, _ in heapq.nlargest(n, data.items(), key=lambda item: item[1])]
            else:
                data = list(data.keys())

            tokens.append(data)


    def encode(self, text: str):

        pass
    
    def decode(self, vect) -> str:

        pass


    def extend_vocabulary(self, new_tokens):

        pass

    def reduce_vocabulary(self, del_tokens):

        pass
