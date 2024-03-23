import math
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (
            self.data[index : index + self.seq_length],
            self.data[index + 1 : index + self.seq_length + 1],
        )
        

def generate_positional_encodings(max_len, d_model):
    pos_encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pos_encodings[:, 0::2] = torch.sin(position * div_term)
    pos_encodings[:, 1::2] = torch.cos(position * div_term)
    return pos_encodings.unsqueeze(0)
