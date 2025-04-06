from torch import nn 
from tokenizer import tokenize


class Embeddings(nn.Module):

    def __init__(
            self,
            d_model: int,                                     # unchangable
            vocab_size: int,                                  # can be changed with an internal funcion
            saved_tokens_path: str | None = None,             # file with saved tokenization
            dataset_path: str | None = None,                  # path to the dataset, makes sense only if vocab_size is defined
            min_token_occurrence: float | None = None,        # min number of average occurances of the token in the dataset per line
            verbose: int = 1                                  # value of 0 or 1 representing whether info is printed    
        ) -> None:

        if verbose not in [0, 1]:
            raise Exception("Verbose must be equal to `0` or `1`!")

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size


        if saved_tokens_path is not None:
            pass
        else:
            tokens = tokenize(
                vocab_size=vocab_size, 
                dataset_path=dataset_path, 
                min_token_occurrence=min_token_occurrence, 
                verbose=verbose
            )
        
        self.tokens = {tokens[i] : i for i in range(len(tokens))}
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.projection = nn.Linear(d_model, vocab_size)

    def encode(self, text: str):

        pass
    
    def decode(self, vect) -> str:

        pass


    def extend_vocabulary(self, new_tokens):

        pass

    def reduce_vocabulary(self, del_tokens):

        pass
