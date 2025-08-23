import numpy as np
import torch
import torch.nn as nn

import pydantic 

from typing import Optional, Tuple

class ModelConfig(pydantic.BaseModel):
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 200
    embeddings_lr: float = 0.001
    weight_decay: float = 1.0



# NOT BEING USED 
# model now uses TransformerBlock and MultiheadedAttention

class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Compute attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / self.hidden_dim ** 0.5
        attn_weights = self.softmax(scores)

        # Apply attention weights to values
        context = torch.bmm(attn_weights, values)
        return context


class RecurrentModule(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.attention = Attention(hidden_dim)

        # going with lstm to simplify understanding 
        self.lstm = nn.LSTM(input_size=input_dim, 
                    hidden_size=hidden_dim,
                    num_layers=num_layers, 
                    batch_first=True, dropout=dropout) 

        # Projection layer is only needed if hidden_dim is different from input_dim
        if hidden_dim != input_dim:
            self.projection = nn.Linear(hidden_dim, input_dim, bias=False)
        else:
            self.projection = nn.Identity()
        
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x, hidden=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            hidden = (h0, c0)

        layer_out, hidden = self.lstm(x, hidden)

        output = self.attention(layer_out)
        
        output = self.projection(output)

        output = self.layer_norm(output + x)
        return output