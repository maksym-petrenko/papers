import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import heapq



def is_english_or_french(char: str) -> bool:
    code = ord(char)
    return (
        (0x0020 <= code <= 0x007E) or
        (0x00A0 <= code <= 0x00FF)
    )

def tokenize(
        vocab_size: int, 
        dataset: str,
        min_token_occurrence: float = 1e-4,
        verbose: int = 1
    ) -> list:

    if verbose:
        print("Loading data...")
    df = pd.read_csv(dataset)
    if verbose:
        print("Data is loaded successfully.")
    min_samples = int(min_token_occurrence * len(df))
    data = defaultdict(int)

    if verbose:
        print("Initiating tokens")
    for _, line in tqdm(df.iterrows(), total=len(df), disable=not bool(verbose)):
        for char in (str(line["en"]) + str(line["fr"])):
            data[char] += 1
    data = {char: n for char, n in data.items() if n >= min_samples}
    
    tokens = list(data.keys())
    if verbose:
        print(f"Created {len(tokens)} char tokens in total")
    tokens = [sorted([token for token in tokens if is_english_or_french(token)])]
    max_token_length = 1

    while sum([len(i) for i in tokens]) < vocab_size - 1:
        max_token_length += 1
        data = defaultdict(int)

        for _, line in tqdm(df.iterrows(), total=len(df), disable=not bool(verbose)):
            en_text = str(line["en"])
            fr_text = str(line["fr"])
            for i in range(len(str(en_text)) - max_token_length):
                data[str(en_text)[i:i + max_token_length]] += 1
            for i in range(len(str(fr_text)) - max_token_length):
                data[str(fr_text)[i:i + max_token_length]] += 1
            
        data = {token: n for token, n in data.items() if n >= min_samples}

        if len(data) + sum([len(i) for i in tokens]) > vocab_size - 1:
            n = vocab_size - 1 - sum([len(i) for i in tokens]) 
            data = [k for k, _ in heapq.nlargest(n, data.items(), key=lambda item: item[1])]
        else:
            data = list(data.keys())

        tokens.append(data)

        if verbose:
            print(f"Total token count: {sum([len(i) for i in tokens])}")

    return tokens

