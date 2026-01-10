import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)*torch.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__(self)
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # (seq_len, d_model)
        pe = torch.zeros(self.seq_len, d_model)
        # (seq_len, 1)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        # 2i*log(10000)/d_model, div_term = (d_model,)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* 
                            (-torch.log(10000)/self.d_model))
        # apply trig, shape (seq_len, d_model)
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)