import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import pydantic 

from typing import Optional

# Base module for lower and higher layers
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

        # going with lstm to simplify understanding 
        self.layers = nn.ModuleList(
            nn.LSTM(input_size=input_dim, hidden_size=input_dim,
                   num_layers=num_layers, batch_first=True, dropout=dropout) for _ in range(num_layers)
        )

        self.projection = nn.Linear(input_dim, input_dim, bias=False)

        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x, hidden=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            hidden = (h0, c0)

        for layer in self.layers:
            x, hidden = layer(x, hidden)

        # should add attention?

        output = self.layer_norm(x)

        output = self.projection(x)

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
            input_dim=self.config.input_dim,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        )

        self.Low_net = RecurrentModule(
            input_dim=self.config.input_dim,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        )

        # Combine and project to latent (hrm latent == output_dim)
        self.layer_norm = nn.LayerNorm(self.config.hidden_dim * 2)  # added
        self.output_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.output_dim)
        )

        # Projections
        self.low_level_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=False)
        self.high_level_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim, bias=False)

    def initialize_hidden_states(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize hidden states for low and high level modules
        z0_L = torch.zeros(batch_size, 1,  self.config.hidden_dim, device=self.device)
        z0_H = torch.zeros(batch_size, 1, self.config.hidden_dim, device=self.device)
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
        for layer in network.layers:
            combined, _ = layer(combined)
        return combined


    def forward(self, x, hidden_states=None):
        # x: (B, 81, input_dim)
        x = self.input_proj(x)
        # Initialize hidden states if not provided
        if hidden_states is None:
            high_level_state, low_level_state = self.initialize_hidden_states(x.shape[0])
        else:
            high_level_state, low_level_state = hidden_states

        with torch.no_grad():
            for step in range(self.total_steps - 1):
                low_level_state = self.level_step(
                    low_level_state, high_level_state, x, self.Low_net, self.low_level_proj
                )

                if(step + 1) % self.T == 0:
                    high_level_state = self.level_step(
                        high_level_state, low_level_state, x, self.High_net, self.high_level_proj
                    )

        # 1 step with gradient
        low_level_state = self.level_step(
            low_level_state, high_level_state, x, self.Low_net, self.low_level_proj
        )
        high_level_state = self.level_step(
            high_level_state, low_level_state, x, self.High_net, self.high_level_proj
        )

        combined = torch.cat([low_level_state, high_level_state], dim=-1)
        combined = self.layer_norm(combined)

        latent = self.output_proj(combined)  # (B, 81, output_dim)
        return latent