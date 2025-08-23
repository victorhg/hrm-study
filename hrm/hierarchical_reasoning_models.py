
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