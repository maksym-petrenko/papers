from networks import Transformers
from tqdm import tqdm
import torch
from torch import nn
import pandas as pd

device = "cuda"

model = Transformers(6, 6, 8, 8, 512, 20000, "tokens.txt").to(device=device)
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
        tgt = model.embeddings.encode(str(line["fr"]))
        
        total_loss = 0

        for i in range(1, len(tgt)):
            tokens = tgt[:i]
            text = "".join([model.embeddings.id_to_token(token) for token in tokens])
            
            probs = model(src, text, train=True)

            expected_probs = torch.zeros(model.vocab_size)
            expected_probs[tgt[i]]
            
            loss = criterion(probs, expected_probs)

            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            
        print("loss:", total_loss / len(tgt))

torch.save(model, "trained.pt")

