import pandas as pd
import torch
from networks import Transformers
from torch import nn
from tqdm import tqdm

if __name__ == "__main__":

    device = "cuda"

    print("Loading tokenizer:")
    model = Transformers(6, 6, 8, 8, 512, 20000, 1024, "tokens.txt").to(device=device)
    print("Loading data:")
    df = pd.read_csv("dataset.csv", names=["en", "fr"])

    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    print("Starting training:")
    for epoch in tqdm(range(1, epochs + 1)):

        for _, line in df.iterrows():

            optimizer.zero_grad()

            src = str(line["en"])
            output_tokens = model.embeddings.tokenize([str(line["fr"])])

            padded_tokens = []
            for sample in output_tokens:
                if len(sample) > model.window - 2:
                    sample = sample[:model.window - 2]
                padded_sample = [1] + sample + [2] + [3] * (model.window - len(sample) - 2)
                padded_tokens.append(padded_sample)
            output_tokens_tensor = torch.tensor(padded_tokens, dtype=torch.long).to(device)

            logits = model([src], output_tokens, device=device)

            loss = criterion(logits.view(-1, model.vocab_size), output_tokens_tensor.view(-1))

            loss.backward()
            optimizer.step()

            print("loss:", float(loss))

    torch.save(model, "trained.pt")
