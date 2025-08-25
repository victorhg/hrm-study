import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import pydantic 

from typing import Optional, Tuple

from hrm.transformer import TransformerModule

class ModelConfig(pydantic.BaseModel):
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 200
    embeddings_lr: float = 0.001
    weight_decay: float = 1.0


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
        self.input_proj =  nn.Identity() #nn.Linear(config.input_dim, config.hidden_dim)

        # embedding for sudoku 
        self.digit_embedding = nn.Embedding(10, config.hidden_dim // 2)
        self.position_embedding = nn.Embedding(81, config.hidden_dim // 2)

        self.High_net = TransformerModule(
            input_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            num_heads=4,
            dropout=self.config.dropout
        )

        self.Low_net = TransformerModule(
            input_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            num_heads=4,
            dropout=self.config.dropout
        )


        # Combine and project to latent (hrm latent == output_dim)
        self.layer_norm = nn.LayerNorm(self.config.hidden_dim)  # added
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
        if input_tensor.dim() == 3:
            #batch_size, seq_len = input_tensor.shape
            # ensure (B, 81)
            input_tensor = input_tensor.unsqueeze(-1)
        
        batch_size, seq_len = input_tensor.shape
        
        #input_tensor = self.input_proj(input_tensor)  # (B, 81, hidden_dim)
        #print(f"After input projection: {input_tensor.shape}")
        digit_embeds = self.digit_embedding(input_tensor.long())  # (B, 81, hidden_dim//2)
        positions = torch.arange(0, 81, device=self.device).unsqueeze(0).expand(batch_size,-1 )
        position_embeds = self.position_embedding(positions)  # (B, 81, hidden_dim//2)

        combined = torch.cat([digit_embeds, position_embeds], dim=-1)  # (B, 81, hidden_dim)

        if hidden_states is None:
            high_level_state, low_level_state = self.initialize_hidden_states(batch_size, seq_len)
        else:
            high_level_state, low_level_state = hidden_states

        with torch.no_grad():
            for step in range(self.total_steps - 1):
                low_level_state = self.level_step(
                    low_level_state, high_level_state, combined, self.Low_net, self.low_level_proj
                )

                if(step + 1) % self.T == 0:
                    high_level_state = self.level_step(
                        high_level_state, low_level_state, combined, self.High_net, self.high_level_proj
                    )

        # last step with gradient
        for t in range(self.T):
            low_level_state = self.level_step(
                low_level_state, high_level_state, combined, self.Low_net, self.low_level_proj
            )

        high_level_state = self.level_step(
                high_level_state, low_level_state, combined, self.High_net, self.high_level_proj
            )
        

        high_level_state = self.layer_norm(high_level_state)

        output = self.output_proj(high_level_state)  # (B, 81, output_dim)
        #print(f"Final output shape: {output.shape}")
        
        return output