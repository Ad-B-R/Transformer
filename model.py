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
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h:int, dropout:int):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout
        assert d_model%h == 0,  "d_model not divisible by h"

        self.d_k = d_model//h
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, seq_len, d_k) -> (batch, seq_len, seq_len)
        attention_scores = ((query @ key.transpose(-2,-1))/torch.sqrt(d_k))
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # batch, h, seq_len, seq_len
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ key), attention_scores


    def forward(self, k, q, v, mask):
        # batch x seq len x d_model
        query = self.Wq(q)
        value = self.Wq(v)
        key = self.Wq(k)
        # batch, seq_len, h, d_k -> batch, h, seq_len, d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose([1,2])
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose([1,2])
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose([1,2])

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        