import torch.nn as nn
from modules.utils import FourierFeatureTransform

class Decoder(nn.Module):
    def __init__(self, depth, width, out_dim, input_dim=3, positional_encoding=False, sigma=5.0):
        super(Decoder, self).__init__()
        
        # mlp
        layers = []
        if positional_encoding == 1:
            layers.append(FourierFeatureTransform(input_dim, width, sigma))
            layers.append(nn.Linear(width * 2 + input_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width]))
        else:
            layers.append(nn.Linear(input_dim, width[0]))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([width[0]]))
        input_dim_temp = width[0]
        for w0 in width[1:]:
            layers.append(nn.Linear(input_dim_temp, w0))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([w0]))
            input_dim_temp = w0  # Update input dimension for subsequent layers

        layers.append(nn.Linear(input_dim_temp, out_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.mlp = nn.ModuleList(layers)
        print(self.mlp)
        
    def forward(self, x):
        # mlp
        for layer in self.mlp:
            x = layer(x)
        return x