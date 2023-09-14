import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers = num_layers,
                            batch_first=False)
        
        self.linear1 = nn.Linear(hidden_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size)
        
        nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
            
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out, hidden
    
    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h0, c0