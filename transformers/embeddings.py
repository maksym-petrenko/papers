from torch import nn 
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

class Embeddings(nn.Module):

    def __init__(
            self,
            d_model: int,                       # unchangable
            vocab_size: int,                    # can be changed with an internal funcion
            dataset: str | None = None          # path to the dataset, makes sense only if vocab_size is defined
        ) -> None:

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_to_id = None
        self.embeddings = None
        self.projection = nn.Linear(d_model, vocab_size)

        if dataset is not None:
            df = pd.read_csv(dataset)
            data = defaultdict(int)
            for _, line in tqdm(df.iterrows(), total=len(df)):
                for char in (str(line["en"]) + str(line["fr"])):
                    data[char] += 1
            data = {char: n for char, n in data.items() if n >= 100}
            print(data)

    def encode(self, text: str):

        pass
    
    def decode(self, vect) -> str:

        pass


    def extend_vocabulary(self, new_tokens):

        pass

    def reduce_vocabulary(self, del_tokens):

        pass
