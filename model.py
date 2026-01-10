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
        # [1, seq_len, d_model]
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    @torch.no_grad()
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :] # for different broadcasting purposes
        return self.dropout(x)
class LayerNorm(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.parameter(torch.ones(1))
        self.beta = nn.parameter(torch.ones(1))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x-mean)/(std + self.eps) + self.beta

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fnn1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fnn2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fnn2(self.dropout(torch.relu(self.fnn1(x))))