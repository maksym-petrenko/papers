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
            output_tokens = model.embeddings.tokenize(str(line["fr"]))

            total_loss = 0

            for i in range(1, len(output_tokens)):
                probs = model(src, output_tokens, train=True)

                expected_probs = torch.zeros(model.vocab_size).to(device=device)
                expected_probs[output_tokens[i]] = 1

                loss = criterion(probs, expected_probs)

                loss.backward()
                optimizer.step()

                total_loss += float(loss)

            print("loss:", total_loss / len(output_tokens))

    torch.save(model, "trained.pt")
