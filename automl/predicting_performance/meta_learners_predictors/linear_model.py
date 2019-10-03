import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.linear = torch.nn.Linear(inputSize, outputSize).to(self.device)
        
    def forward(self, x):
        return self.linear(x)