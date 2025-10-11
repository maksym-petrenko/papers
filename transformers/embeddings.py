import torch
from tokenizer import read_tokens, tokenize
from torch import nn


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
            return TrieNode("<UNK>", 1)
        return node

    def add_token(self, token, index):

        best = self.find_best_match(token)
        to_add = token[len(best.token) :]

        if to_add == best.token:
            best.index = index
            return None

        for char in to_add:
            idx = index if to_add == token else None
            best.children[char] = TrieNode(to_add + char, idx)


class Embeddings(nn.Module):

    def __init__(
        self,
        d_model: int,  # unchangeble
        vocab_size: int,  # can be changed with an internal funcion
        dataset_path: (
            str | None
        ) = None,  # path to the dataset, makes sense only if vocab_size is defined
        saved_tokens_path: str | None = None,  # file with saved tokenization
        min_token_occurrence: float = 1e-6,  # min number of average occurances of the token in the dataset per line
        verbose: int = 1,  # value of 0 or 1 representing whether info is printed
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
                raise Exception(
                    "input must contain either dataset path or saved tokens path!"
                )
            tokens = tokenize(
                vocab_size=vocab_size,
                dataset_path=dataset_path,
                min_token_occurrence=min_token_occurrence,
                verbose=verbose,
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
        text: list[str] | list[list[int]],
        window: int | None = None,
    ) -> list | torch.Tensor:

        batch = len(text)

        tokens = text
        if isinstance(text, str):
            tokens = self.tokenize(text)

        if window is not None:
            tokens = [
                [sample[i] if i < len(sample) else 3 for i in range(window - 2)]
                for sample in tokens
            ]
            tokens = [[1] + sample + [2] for sample in tokens]
            tokens = torch.tensor(tokens, dtype=torch.int)
            print(tokens.size())

            tokens = torch.cat(
                (torch.ones(batch, window), tokens, 2 * torch.ones(batch, window))
            ).int()
            return torch.cat([self.embeddings[sample] for sample in tokens], dim=0)

        return tokens

    def decode(self, text, return_probabilities=False) -> str:

        proj = self.projection(text)
        if return_probabilities:
            return proj

        # batches unnecessary here yet
        tokenids = [torch.argmax(proj[i]) for i in range(len(text))]
        text = ""

        for tokenid in tokenids:
            if tokenid == 2:
                break
            text += self.id_to_token[tokenid]

        return text

    def tokenize(self, text: list[str]) -> list[int]:

        result = []
        for line in text:
            tokens = []

            while line:
                best = self.tokens.find_best_match(line)

                if best.token == "<UNK>" or (not best.token) or best.index is None:
                    line = text[1:]
                    tokens.append(1)
                else:
                    line = text[len(best.token) :]
                    tokens.append(best.index)
            result.append(tokens)

        return result
