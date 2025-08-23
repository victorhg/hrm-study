import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import pydantic 

from typing import Optional, Tuple

# Base module for lower and higher layers


class ModelConfig(pydantic.BaseModel):
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 200
    embeddings_lr: float = 0.001
    weight_decay: float = 1.0


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
    
# ####################
# HierarchicalReasoningModel
 # added to ensure HRMConfig is defined successfully
class HRMConfig(pydantic.BaseModel):
    input_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    output_dim: int = 10

    N: int = 2  # number of high-level module cycles
    T: int = 4  # number of low-level module cycles
    max_seq_len: int = 256

class HierarchicalReasoningModel(nn.Module):
    def __init__(self, config: HRMConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.total_steps = config.N * config.T  # total steps in the HRM
        self.device = device
        self.N = config.N
        self.T = config.T
        # Define model layers here

        # Input projection (project puzzle embedding dim -> hidden)
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        self.High_net = RecurrentModule(
            input_dim=self.config.hidden_dim,  # Use hidden_dim, not input_dim
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        )

        self.Low_net = RecurrentModule(
            input_dim=self.config.hidden_dim,  # Use hidden_dim, not input_dim
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        )

        # Combine and project to latent (hrm latent == output_dim)
        self.layer_norm = nn.LayerNorm(self.config.hidden_dim * 2)  # added
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # Projections
        self.low_level_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=False)
        self.high_level_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=False)

    def initialize_hidden_states(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize hidden states for low and high level modules
        z0_L = torch.zeros(batch_size, seq_len, self.config.hidden_dim, device=self.device)
        z0_H = torch.zeros(batch_size, seq_len, self.config.hidden_dim, device=self.device)
        return z0_H, z0_L


    def level_step(self,
                   first_level: torch.Tensor,
                   second_level: torch.Tensor,
                   input_embedding: torch.Tensor,
                   network: nn.Module,
                   projection: nn.Module
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        level_influence = projection(second_level)
        combined = first_level + level_influence + input_embedding
        combined = network(combined)
        return combined


    def forward(self, input_tensor, hidden_states=None):
        # Handle both 2D and 3D inputs
        if input_tensor.dim() == 2:
            batch_size, seq_len = input_tensor.shape
            # Add feature dimension: (B, 81) -> (B, 81, 1)
            input_tensor = input_tensor.unsqueeze(-1)
        else:
            batch_size, seq_len, _ = input_tensor.shape
        
        # Project input to hidden dimension
        input_tensor = self.input_proj(input_tensor)  # (B, 81, hidden_dim)
        #print(f"After input projection: {input_tensor.shape}")
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            high_level_state, low_level_state = self.initialize_hidden_states(batch_size, seq_len)
        else:
            high_level_state, low_level_state = hidden_states

        with torch.no_grad():
            for step in range(self.total_steps - 1):
                low_level_state = self.level_step(
                    low_level_state, high_level_state, input_tensor, self.Low_net, self.low_level_proj
                )

                if(step + 1) % self.T == 0:
                    high_level_state = self.level_step(
                        high_level_state, low_level_state, input_tensor, self.High_net, self.high_level_proj
                    )

        # 1 step with gradient
        low_level_state = self.level_step(
            low_level_state, high_level_state, input_tensor, self.Low_net, self.low_level_proj
        )
        high_level_state = self.level_step(
            high_level_state, low_level_state, input_tensor, self.High_net, self.high_level_proj
        )

        # Project high-level state to output classes
        output = self.output_proj(high_level_state)  # (B, 81, output_dim)
        #print(f"Final output shape: {output.shape}")
        
        return output