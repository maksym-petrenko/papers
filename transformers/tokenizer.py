import concurrent.futures
import heapq
import platform
import subprocess
import time
from collections import Counter

import pandas as pd


def is_english_or_french(line: str) -> bool:
    for char in line:
        code = ord(char)
        if not ((0x0020 <= code <= 0x007E) or (0x00A0 <= code <= 0x00FF)):
            return False
    return True


def load_chunk(dataset_path: str, start_row: int, finish_row: int) -> pd.DataFrame:

    return pd.read_csv(
        dataset_path,
        skiprows=start_row,
        nrows=finish_row - start_row,
        names=["en", "fr"],
    )


def process_chunk(df: pd.DataFrame, token_length: int) -> dict:

    local_data = Counter()

    for _, line in df.iterrows():
        en_data = str(line["en"])
        fr_data = str(line["fr"])

        words = en_data.split() + fr_data.split()

        for word in words:
            for i in range(len(word) - token_length):
                local_data[word[min(i, 1) : (i + token_length)]] += 1

    local_data = {
        token: count
        for token, count in local_data.items()
        if is_english_or_french(token)
    }

    return local_data


def tokenize(
    vocab_size: int,
    dataset_path: str,
    min_token_occurrence: float = 1e-4,
    num_workers: int = 0,
    verbose: int = 1,
) -> list:

    if num_workers == 0:
        import multiprocessing

        num_workers = multiprocessing.cpu_count() - 4

    if platform.system() == "Windows":
        lines = 0
        with open(dataset_path, "r", encoding="utf-8") as f:
            for _ in f:
                lines += 1
    else:
        result = subprocess.run(
            ["wc", "-l", dataset_path], capture_output=True, text=True, check=True
        )
        output_line = result.stdout.strip()
        lines = int(output_line.split()[0])

    min_samples = int(lines * min_token_occurrence)
    chunk_size = lines // num_workers
    token_length = 1
    data = dict()

    dfs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            start_row = i * chunk_size
            finish_row = (i + 1) * chunk_size if i < num_workers - 1 else lines
            futures.append(
                executor.submit(load_chunk, dataset_path, start_row, finish_row)
            )

        for future in concurrent.futures.as_completed(futures):
            dfs.append(future.result())

    while len(data) < vocab_size - 4:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            futures = []
            for i in range(num_workers):
                futures.append(executor.submit(process_chunk, dfs[i], token_length))

            for future in concurrent.futures.as_completed(futures):
                data.update(future.result())

        data = {token: count for token, count in data.items() if count >= min_samples}

        token_length += 1

    data = heapq.nlargest(vocab_size, data.items(), key=lambda item: item[1])
    tokens = ["<UNK>", "<SOS>", "<EOS>", "<PAD>", " "] + [token[0] for token in data]

    return tokens


def write_tokens(tokens: list[str], path: str) -> None:

    with open(path, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(f"{token}\n")


def read_tokens(path: str) -> list[str]:

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", help="path to save the tokens to", type=str)
    parser.add_argument("-d", "--dataset", help="path to the dataset", type=str)
    parser.add_argument("-vocab", "--vocab", help="vocabulary size", type=int)
    parser.add_argument("-d_model", "--d_model", help="embeddings size", type=int)
    parser.add_argument(
        "-min_occurrence", "--min_occurrence", help="self explanatory", type=float
    )

    args = parser.parse_args()

    path = args.path
    dataset_path = args.dataset

    if path is None:
        path = "tokens.txt"

    tokens = tokenize(
        vocab_size=args.vocab,
        dataset_path=dataset_path,
        min_token_occurrence=args.min_occurrence,
    )

    write_tokens(tokens, path)
