import torch
from torch import nn
import numpy as np 

class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, is_train: bool = True):
        super().__init__()
        
        self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1))
        self.input_size, self.output_size = input_size, output_size
        
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k+1) / g
        
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])), requires_grad=is_train).to('cuda')
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])), requires_grad=is_train).to('cuda')
        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size)).to('cuda')

        
    def forward(self, x):
        x = x.unsqueeze(1).expand(-1, self.k+self.g)
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_height - x)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        x = x.reshape((self.output_size, 1))
        x = x.squeeze()
        return x


class ReLUKAN(nn.Module):

    def __init__(self, dimensions, g, k):
        super().__init__()

        self.layers = []
        self.g = g 
        self.k = k

        for i in range(len(dimensions) - 1):
            self.layers.append(ReLUKANLayer(dimensions[i], g, k, dimensions[i+1]).to('cuda'))
    
    def parameters(self):
        for layer in self.layers:
            for param in layer.parameters():
                yield param

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
