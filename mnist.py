import torch
from torch import nn
from kan import ReLUKAN
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader


class ReLUKAN_MNIST(nn.Module):

    def __init__(self):
        
        super().__init__()
        self.kan = ReLUKAN([28*28, 128, 32, 10], 5, 3)
        
    def forward(self, x):

        x = x.flatten()
        x = self.kan(x)
        x = nn.Softmax()(x)
        return x

model = ReLUKAN_MNIST()
model = model.to('cuda')
print(next(model.parameters()).device)


transform = Compose([ToTensor()])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)

        one_hot = torch.zeros(10, dtype=torch.float32).to('cuda')
        one_hot[labels[0]] = 1

        loss = criterion(outputs, one_hot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on the training set: {100 * correct / total}%')
