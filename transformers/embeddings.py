import torch
from torch import nn 


class Embeddings(nn.Module):

    def __init__(
            self,
            d_model: int,                       # unchangable
            vocab_size: int | None = None,      # can be changed with an internal funcion
            dataset: str | None = None          # path to the dataset, makes sense only if vocab_size is defined
        ) -> None:

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_to_id = None
        self.embeddings = None
        self.projection = None

        if dataset is not None and vocab_size is not None:
            # TODO funciton to iniciate the embeddings
            pass



    def encode(self, text: str) -> torch.Tensor:

        pass
    
    def decode(self, vect: torch.Tensor) -> str:

        pass


    def extend_vocabulary(self, new_tokens):

        pass

    def reduce_vocabulary(self, del_tokens):

        pass
