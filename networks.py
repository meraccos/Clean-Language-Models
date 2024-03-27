import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import generate_positional_encodings

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.output = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, mask):
        tok_enc = self.tok_embedding(trg) * self.scale
        pos_enc = self.pos_embedding(torch.arange(0, trg.shape[-1]).to(self.device))
        x = self.dropout(tok_enc + pos_enc)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.output(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        ''' Multi Head self Attention'''
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, device)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        ''' Encoder-decoder attention'''
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, device)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)

        ''' Positionwise FeedForward Layer'''
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, mask):
        # x = self.dropout(trg)
        x = trg
        att, _ = self.self_attention(x, x, x, mask = mask)
        att = self.dropout(att)
        x = self.self_attn_layer_norm(att + x)
        # att, attention = self.encoder_attention(x, enc_src, enc_src, mask=src_mask)
        # att = self.dropout(att)
        # x = self.enc_attn_layer_norm(att + x)
        ff = self.positionwise_feedforward(x)
        ff = self.dropout(ff)
        x = self.ff_layer_norm(ff + x)
        return x


class Attention_Network(nn.Module):
    def __init__(self, n_vocab, embd, nhead, n_layers, ff_dim, max_len):
        super().__init__()
        self.embd = embd
        self.tok_emb = nn.Embedding(n_vocab, embd)
        self.pos_emb = nn.Embedding(max_len, embd)
        
        enc_layer = nn.TransformerEncoderLayer(d_model=embd,
                                               nhead=nhead,
                                               dropout=0.0)
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer,
                                             num_layers=n_layers,
                                             norm=nn.LayerNorm(embd)
                                             )
        
        self.linear1 = nn.Linear(embd, ff_dim)
        self.linear2 = nn.Linear(ff_dim, n_vocab)
        # self.pe = PositionalEncoding(embd)
    
    def forward(self, x):
        pos = torch.arange(0, x.shape[1])
        tok_emb = self.tok_emb(x) * self.embd ** 0.5
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        m = nn.Transformer.generate_square_subsequent_mask(100)
        m = None
        # m = self.get_mask(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x, mask=m)
        x = x.permute(1, 0, 2)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        
        return x
    
    def get_mask(self, x):
        mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float('-inf')), diagonal=1)
        return mask


class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 200):
        super().__init__()

        self.device = device
        
        ''' Input Embedding '''
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        ''' Multiple Encoder Layers '''
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
        self.linear = nn.Linear(hid_dim, input_dim)
        
    def forward(self, src, src_mask):
        tok_enc = self.tok_embedding(src) * self.scale
        pos_enc = self.pos_embedding(torch.arange(0, src.shape[-1]).to(self.device))
        x = self.dropout(tok_enc + pos_enc)
        
        for layer in self.layers:
            x = layer(x, src_mask)
        
        x = self.linear(x)
        return x       
    

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        ''' Multi Head self-Attention '''        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, device)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        ''' Positional FeedForward Layer'''
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask):
        att, _ = self.self_attention(x, x, x, src_mask)
        att = self.dropout(att)
        x = self.self_attn_layer_norm(att + x)
        ff = self.positionwise_feedforward(x)
        ff = self.dropout(ff)
        x = self.ff_layer_norm(ff + x)
        return x

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, device):
        super().__init__()
        self.n_heads = n_heads
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.w_cat = nn.Linear(hid_dim, hid_dim)
        
    def forward(self, q, k, v, mask = None):
        (B, N, C), H = k.shape, self.n_heads
        d_tensor = C // H

        q = self.w_q(q).view(B, q.shape[1], H, d_tensor).transpose(1, 2)
        k = self.w_k(k).view(B, k.shape[1], H, d_tensor).transpose(1, 2)
        v = self.w_v(v).view(B, v.shape[1], H, d_tensor).transpose(1, 2)

        scores = q @ k.transpose(2, 3) / d_tensor ** 0.5
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -100000)
            
        attention = F.softmax(scores, dim=-1)
        v = attention @ v 
        out = v.transpose(1, 2).contiguous().view(B, v.shape[2], C)
        out = self.w_cat(out)
        return out, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(hid_dim, pf_dim)
        self.linear2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class LSTM_Network(nn.Module):
    def __init__(self, n_vocab, hidden_size, embd, n_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_vocab, embd)
        self.lstm = nn.LSTM(
            input_size=embd,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            # dropout=0.3
        )

        self.linear = nn.Linear(hidden_size, n_vocab)

        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.linear(out)
        return out, hidden

class Bigram_Network(nn.Module):
    def __init__(self, num_emb, embd):
        super().__init__()
           
        self.emb = nn.Embedding(num_emb, embd)
        self.linear1 = nn.Linear(embd, num_emb)
        self.linear2 = nn.Linear(num_emb, num_emb)
    def forward(self, x):
        x = self.emb(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    