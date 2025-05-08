from networks import Transformers
from tqdm import tqdm
import torch
import pandas as pd

model = Transformers(6, 6, 8, 8, 512, 20000, "tokens.txt")
df = pd.read_csv(
    "dataset.csv",
    names=["en", "fr"]
)

epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in tqdm(range(1, epochs + 1)):
    
    for _, line in df.iterrows():

        # put training here
        
        ...

torch.save(model, "trained.pt")

