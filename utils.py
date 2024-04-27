import os
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class CharDatasetFromText(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    # def preprocess(self):
        

    def __len__(self):
        return len(self.data) - self.block_size + 1

    def __getitem__(self, index):
        return self.data[index : index + self.block_size]


class CharDatasetFromFolder(Dataset):
    def __init__(self, dataset, block_size, split, train_portion=0.98):
        self.dataset = dataset
        self.folder_path = os.path.join('datasets', dataset)
        self.block_size = block_size
        self.split = split
        self.train_portion = train_portion
        self.paths = sorted(os.listdir(self.folder_path))
        self.preprocess()

    def preprocess(self):
        # Read the text files
        texts = []
        for path in self.paths:
            with open(os.path.join(self.folder_path, path), "r") as f:
                texts.append(f.read())
        
        # Build the vocabulary
        self.vocab = sorted(list(set(''.join(texts))))
        self.n_vocab = len(self.vocab)

        # Tokenization
        stoi = {ch: i for i, ch in enumerate(self.vocab)}
        itos = {i: ch for i, ch in enumerate(self.vocab)}

        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: "".join(itos[i] for i in l)
        
        # Save the encodings for further decoding
        with open(f'encodings/enc_{self.dataset}', 'w') as f:
            f.write(''.join(self.vocab))

        # Filter out the texts that are shorter than the block size
        texts = [text for text in texts if len(text) > self.block_size]
        
        # Split the dataset
        texts = sorted(texts)
        border_idx = int(len(texts) * self.train_portion)
        texts = texts[:border_idx] if self.split == 'train' else texts[border_idx:]

        # Tokenize the texts
        self.texts = [self.encode(text) for text in texts]
        self.lengs = [len(text) - self.block_size + 1 for text in self.texts]
        self.csums = np.cumsum(self.lengs)

    def __len__(self):
        return sum(self.lengs)-1

    def __getitem__(self, idx):
        file_idx = np.argmax(self.csums > idx)
        text_idx = idx if file_idx == 0 else idx - self.csums[file_idx - 1]
        
        text = self.texts[file_idx]
        block = text[text_idx: text_idx + self.block_size] 
        block_oh = F.one_hot(torch.tensor(block), num_classes=self.n_vocab)
        return block_oh
    

def generate_positional_encodings(max_len, d_model):
    pos_encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pos_encodings[:, 0::2] = torch.sin(position * div_term)
    pos_encodings[:, 1::2] = torch.cos(position * div_term)
    return pos_encodings.unsqueeze(0)

def get_writer(model_name, dataset):
    now = datetime.now().strftime("%b%d_%H-%M")
    return SummaryWriter(f'runs/{now}_{dataset[0]}_{model_name}')
