import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import generate_positional_encodings


class Attention_Network(nn.Module):
    def __init__(self, n_vocab, embd, nhead, n_layers, ff_dim, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(n_vocab, embd)
        self.pos_emb = nn.Embedding(max_len, embd)
        
        enc_layer = nn.TransformerEncoderLayer(d_model=embd,
                                               nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer,
                                             num_layers=n_layers)
        
        self.linear1 = nn.Linear(embd, ff_dim)
        self.linear2 = nn.Linear(ff_dim, n_vocab)
    
    def forward(self, x):
        pos = torch.arange(0, x.shape[1])
        tok_emb = self.tok_emb(x) 
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        m = self.get_mask(x)
        x = self.encoder(x, mask=m)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        
        return x
    
    def get_mask(self, x):
        mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float('-inf')), diagonal=1)
        return mask
        


class LSTM_Network(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # dropout=0.3
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.linear(out)
        return out, hidden
