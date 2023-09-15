import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramModel(nn.Module):
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


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers = num_layers,
                            batch_first=True)
        
        self.linear = nn.Linear(hidden_size, output_size)
        
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
            
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.linear(out)
        return out, hidden
    
    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h0, c0
