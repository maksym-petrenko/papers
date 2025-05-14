import torch
from torch import nn
from tokenizer import tokenize, read_tokens


class TrieNode:

    def __init__(self, token, index):

        self.token = token
        self.index = index
        self.children = dict()

    def find_child(self, char):
        
        return self.children.get(char)
    
    def find_best_match(self, string):

        previous_node = self
        node = self

        for char in string:
            previous_node = node
            node = previous_node.find_child(char)
            if node is None: 
                return previous_node
        if node.index is None or node.token == "":
            return TrieNode('<UNK>', 1)
        return node
    
    def add_token(self, token, index):

        best = self.find_best_match(token)
        to_add = token[len(best.token):] 

        if to_add == best.token:
            best.index = index
            return None

        for char in to_add:
            idx = index if to_add == token else None
            best.children[char] = TrieNode(to_add + char, idx)


class Embeddings(nn.Module):

    def __init__(
            self,
            d_model: int,                                     # unchangeble
            vocab_size: int,                                  # can be changed with an internal funcion
            dataset_path: str | None = None,                                # path to the dataset, makes sense only if vocab_size is defined
            saved_tokens_path: str | None = None,             # file with saved tokenization
            min_token_occurrence: float = 1e-6,               # min number of average occurances of the token in the dataset per line
            verbose: int = 1                                  # value of 0 or 1 representing whether info is printed
        ) -> None:

        if verbose not in [0, 1]:
            raise Exception("Verbose must be equal to `0` or `1`!")

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size


        if saved_tokens_path is not None:
            tokens = read_tokens(saved_tokens_path)
        else:
            if dataset_path is None:
                raise Exception("input must contain either dataset path or saved tokens path!")
            tokens = tokenize(
                vocab_size=vocab_size,
                dataset_path=dataset_path,
                min_token_occurrence=min_token_occurrence,
                verbose=verbose
            )

        self.tokens = TrieNode("", None)
        self.id_to_token = tokens
        self.tokens.add_token(" ", 0)
        for i, token in enumerate(tokens[1:]):
            self.tokens.add_token(token, i)
        self.embeddings = torch.rand(vocab_size, d_model)
        self.projection = nn.Linear(d_model, vocab_size)

    def encode(
            self, 
            text: str, 
            window: int | None = None,
        ):
        if window is not None:
            result = torch.zeros(window, self.d_model)
        else:
            result = torch.zeros(1, self.d_model)
        result[0] = self.embeddings[2]  # add <SOS>
        i = 1
       
        tokens = self.tokenize(text)

        for token in tokens:
            if window is None:
                result = torch.cat((result, torch.zeros(1, self.d_model)), 0)
            else:
                if i >= window - 1:
                    result[-1] = self.embeddings[3]
                    return result

            result[i] = self.embeddings[token]
            i += 1

        if window is None:
            result = torch.cat((result, torch.zeros(1, self.d_model)), 0)
        
        if window is not None:
            if i == window - 1:
                result[i] = self.embeddings[3]  # add <EOS>
                i += 1
            while i < window:
                result[i] = self.embeddings[4]  # add <PAD>
                i += 1

        return result

    def decode(self, text, return_probabilities=False) -> str:

        proj = self.projection(text)
        if return_probabilities:
            return proj

        tokenids = [torch.argmax(proj[i]) for i in range(len(text))]
        text = ""

        for tokenid in tokenids:
            if tokenid == 2:
                break
            text += self.id_to_token[tokenid]
        
        return text

    def tokenize(self, text) -> list[int]:

        tokens = []

        while text:
            best = self.tokens.find_best_match(text)

            if best.token == "<UNK>" or (not best.token) or best.index is None:
                text = text[1:]
                tokens.append(1)
            else:
                text = text[len(best.token):]
                tokens.append(best.index)
        return tokens

