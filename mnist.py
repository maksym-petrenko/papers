import torch
from torch import nn
from kan import ReLUKAN
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, random_split


class ReLUKAN_MNIST(nn.Module):

    def __init__(self):
        
        super().__init__()
        self.kan = ReLUKAN([28**2, 128, 32, 10], 5, 3)
        
    def forward(self, x):

        x = x.reshape(x.shape[0], -1)
        x = self.kan(x)
        x = nn.Softmax(dim=-1)(x)
        
        return x

model = ReLUKAN_MNIST()
model = model.to('cuda')

transform = Compose([ToTensor()])
dataset = MNIST(root='./data', train=True, download=True, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_subset, val_subset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=1024, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

verbose = True
num_epochs = 1000

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)

        one_hot = nn.functional.one_hot(labels, num_classes=10).to(dtype=torch.float32)

        loss = criterion(outputs, one_hot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    if verbose:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')
    
    if not (epoch + 1) % 10:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                predicted = torch.argmax(outputs.data, -1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print(f'Accuracy on the validation set: {100 * correct / total}%')
