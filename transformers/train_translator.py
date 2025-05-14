from networks import Transformers
from tqdm import tqdm
import torch
from torch import nn
import pandas as pd

device = "cuda"

model = Transformers(6, 6, 8, 8, 512, 20000, 1024, "tokens.txt").to(device=device)
df = pd.read_csv(
    "dataset.csv",
    names=["en", "fr"]
)

epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

for epoch in tqdm(range(1, epochs + 1)):
    
    for _, line in df.iterrows():

        optimizer.zero_grad()

        src = str(line["en"])
        output_tokens = [2] + model.embeddings.tokenize(str(line["fr"]))

        total_loss = 0

        for i in range(1, len(output_tokens)):
            tokens = output_tokens[:i]
            text = "".join([model.embeddings.id_to_token[token] for token in tokens])

            probs = model(src, text, train=True)

            expected_probs = torch.zeros(model.vocab_size).to(device=device)
            expected_probs[output_tokens[i]] = 1
    
            loss = criterion(probs, expected_probs)

            loss.backward()
            optimizer.step()

            total_loss += float(loss)

        print("loss:", total_loss / len(output_tokens))

torch.save(model, "trained.pt")

