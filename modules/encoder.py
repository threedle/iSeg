import torch.nn as nn
from modules.utils import FourierFeatureTransform

class Encoder(nn.Module):
    def __init__(self, depth, width, out_dim, input_dim=3, positional_encoding=False, sigma=5.0, network_verbose=1):
        super(Encoder, self).__init__()
        layers = []
        if positional_encoding:
            layers.append(FourierFeatureTransform(input_dim, width, sigma))
            layers.append(nn.Linear(width * 2 + input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        else:
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        layers.append(nn.Linear(width, out_dim))
        layers.append(nn.Tanh())
        
        self.mlp = nn.ModuleList(layers)
        if network_verbose == 1:
            print(self.mlp)
    
    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x