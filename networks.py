import torch.nn as nn
import torch.nn.functional as F


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
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        # out = F.dropout(out, p=0.5)
        out = self.linear(out)
        return out, hidden
