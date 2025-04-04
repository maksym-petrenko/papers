import pandas as pd
from collections import defaultdict
from pandas.core.arrays.timedeltas import truediv_object_array
from tqdm import tqdm
import heapq
import concurrent.futures
from collections import Counter
import platform
import subprocess


def is_english_or_french(line: str) -> bool:
    flag = True 
    for char in line:
        code = ord(char)
        if not (
            (0x0020 <= code <= 0x007E) or
            (0x00A0 <= code <= 0x00FF)
        ):
            return False
    return True


def process_chunk(
        dataset: str, 
        start_row: int,
        finish_row: int,
        token_length: int
    ) -> dict:

    df = pd.read_csv(
        dataset, 
        skiprows=start_row,
        nrows=finish_row - start_row
    )

    local_data = Counter()

    for _, line in df.iterrows():
        en_data = str(line["en"])
        fr_data = str(line["fr"])

        words = en_data.split() + fr_data.split()

        for word in words:
            for i in range(len(word) - token_length):
                local_data[word[i:(i + token_length)]] += 1

    local_data = {token: count for token, count in data.items() if is_english_or_french(token)}

    return local_data 


def tokenize(
        vocab_size: int, 
        dataset: str,
        min_token_occurrence: float = 1e-4,
        num_workers: int = 0,
        verbose: int = 1
    ) -> list:

    if num_workers == 0:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()

    data = Counter()
    if platform.system() == "Windows":
        lines = 0
        with open(dataset, 'r', encoding='utf-8') as f:
            for _ in f:
                lines += 1
    else:
        result = subprocess.run(
            ['wc', '-l', dataset],
            capture_output=True,
            text=True,
            check=True
        )
        output_line = result.stdout.strip()
        lines = int(output_line.split()[0])

    min_samples = int(lines * min_token_occurrence)
    chunk_size = lines // num_workers
    token_length = 1

    while len(data) < vocab_size - 2:

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                start_row = i * chunk_size
                finish_row = (i + 1) * chunk_size if i < num_workers - 1 else lines
                futures.append(executor.submit(process_chunk, dataset, start_row, finish_row, token_length))
            
            for future in concurrent.futures.as_completed(futures):
                data.update(future.result())

            data = {token: count for token, count in data.items() if count >= min_samples}

    data = heapq.nlargest(vocab_size, data.items(), key=lambda item: item[1])
    tokens = [" ", "undefined"] + list(data.keys())

    return tokens

