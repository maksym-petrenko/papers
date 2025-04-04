import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import heapq
import concurrent.futures
from collections import Counter
from functools import partial
import platform
import subprocess


def is_english_or_french(char: str) -> bool:
    code = ord(char)
    return (
        (0x0020 <= code <= 0x007E) or
        (0x00A0 <= code <= 0x00FF)
    )


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

    return local_data 


def tokenize(
        vocab_size: int, 
        dataset: str,
        min_token_occurrence: float = 1e-4,
        num_workers: int | None = None,
        verbose: int = 1
    ) -> list:

    if num_workers is None:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()

    data = Counter()
    if platform.system() == "Windows":
        pass
    result = subprocess.run(
        ['wc', '-l', dataset],
        capture_output=True,
        text=True,
        check=True
    )
    output_line = result.stdout.strip()
    chunk_size = int(output_line.split()[0])
    token_length = 1

    process_func = partial(process_chunk, dataset=dataset)
 

    while len(data) < vocab_size - 2:

        futures = []
        for i in range(num_workers):
            start_row = i * chunk_size
            finish_row = (i + 1) * chunk_size if i < num_workers - 1 else total_rows
            futures.append(executor.submit(process_chunk, dataset_path, start_row, finish_row, token_length))
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            data.update(future.result())
   




    
    tokens = list(data.keys())
    if verbose:
        print(f"Created {len(tokens)} char tokens in total")
    tokens = [sorted([token for token in tokens if is_english_or_french(token)])]
    max_token_length = 1

    while sum([len(i) for i in tokens]) < vocab_size - 2:
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

