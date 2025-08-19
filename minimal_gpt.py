import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and split into heads
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        
        # Concatenate heads and put through final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward with residual connection and layer norm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_causal_mask(self, seq_len):
        """Create a causal (lower triangular) mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(x.device)
        
        # Token embeddings + positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
            
        # Final layer norm and output projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        
        return logits

# Example usage and simple training loop
def generate_text(model, start_tokens, max_length=50, temperature=1.0):
    """Simple text generation function"""
    model.eval()
    tokens = start_tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokens)
            next_token_logits = logits[0, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
    return tokens

if __name__ == "__main__":
    # Model parameters
    vocab_size = 10000
    d_model = 256
    n_heads = 8
    n_layers = 4
    d_ff = 1024
    max_len = 512
    
    # Create model
    model = MinimalGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example forward pass
    batch_size = 2
    seq_len = 100
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Example generation
    start_tokens = torch.randint(0, vocab_size, (1, 10))
    generated = generate_text(model, start_tokens, max_length=20)
    print(f"Generated sequence shape: {generated.shape}")
