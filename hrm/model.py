from hrm.transformer import TransformerBlock
from hrm.layers import FeedForward, PositionalEncoding

# ChatGPT 2 model parameteres
GPT_CONFIG_124M = {
        "name": "gpt2-small",
        "vocab_size": 50257,
        "context_length": 1024, 
        "emb_dim": 768, 
        "n_heads": 12, 
        "n_layers": 12, 
        "drop_rate": 0,
        "drop_shortcut": 0.1, 
        "drop_embedding": 0.1,
        "drop_attention": 0.1,
        "qkv_bias": False  
}

class GPTModel(nn.Module):
    def __init__(self, cfg=GPT_CONFIG_124M):
        super().__init__()
        # define embeddings 
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_embedding"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = layers.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb( torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits

    def generate(self, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None, device="cpu"):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self(idx_cond).to(device)
            logits = logits[:, -1, :]
            if top_k is not None:  
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:,  -1]
                logits = torch.where( logits < min_val,
                    torch.tensor(float('-inf')).to(logits.device),
                    logits
                )
            if temperature > 0.0: 
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            if idx_next == eos_id: 
                break

            idx = torch.cat((idx, idx_next), dim=1)
        return idx