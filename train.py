import os
import glob
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data

from model import LSTM


class TextDataset(data.Dataset):
    def __init__(self, chars, v_idx, n_vocab, seq_length):
        self.chars = chars
        self.v_idx = v_idx
        self.n_vocab = n_vocab
        self.seq_length = seq_length

    def __len__(self):
        return len(self.chars) - self.seq_length - 1

    def __getitem__(self, index):
        x_str = self.chars[index:index+self.seq_length]
        y_str = self.chars[index+1:index+self.seq_length+1]
        
        x = torch.tensor([self.v_idx[char] for char in x_str], dtype=torch.long)
        y = torch.tensor([self.v_idx[char] for char in y_str], dtype=torch.long)
        return x, y


class Nizami:
    def __init__(self, model_path = None):
        self.n_hidden = 128
        self.batch_size = 64
        self.block_size = 32
        self.num_layers = 1
        self.lr = 0.0001
        
        self.step = 0
        self.eval_freq = 200
        self.eval_batches = 20
        self.model_save_freq = 5000
        self.process_books()
        self.prepare_model(model_path)
        
    def prepare_model(self, model_path=None):
        self.model = LSTM(self.n_vocab, self.n_hidden, 
                          self.n_vocab, self.num_layers)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def process_books(self):  
        with open('books/nizami.txt', 'r') as f:
            text = f.read()

        self.chars = list(text)
        v = sorted(list(set(self.chars)))
        self.n_vocab = len(v)

        stoi = {ch:i for i, ch in enumerate(v)}
        self.itos = {i:ch for i, ch in enumerate(v)}
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: ''.join(self.itos[i] for i in l)
        
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        x = F.one_hot(x, num_classes=self.n_vocab).type(torch.float32)
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y
    
    def save_model(self):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        file_name = f'models/model_{self.step // self.model_save_freq}.pt'
        torch.save(self.model.state_dict(), file_name)
        
    def generate(self, n_gen_chars):
        char0 = torch.randint(self.n_vocab, (1,))
        x = F.one_hot(char0, self.n_vocab).type(torch.float32)
        for _ in range(n_gen_chars):
            x, (h, c) = self.model(x, None)
            x = F.softmax(x, dim=1)
            char_idx = torch.multinomial(x[0], 1)[0]
            char = self.itos[char_idx.item()]
            print(char, end='')
    
    @torch.no_grad()
    def evaluate(self):
        losses = 0.0
        for _ in range(self.eval_batches):
            x, y = self.get_batch('test')
            
            output, (h, c) = self.model(x, None)
            output = output.view(-1, self.n_vocab)
            y = y.view(-1)
            losses += F.cross_entropy(output, y).detach()
        
        self.writer.add_scalar('Eval/loss', losses / self.eval_batches, self.step)
        
    def train(self, max_epoch):    
        self.writer = SummaryWriter()    
        for epoch in tqdm(range(max_epoch)):
            for _ in range(len(self.chars) // self.batch_size):
                x, y = self.get_batch('train')
                
                output, (h, c) = self.model(x, None)
                output = output.view(-1, self.n_vocab)
                y = y.view(-1)
                loss = F.cross_entropy(output, y)
                
                self.writer.add_scalar('Train/loss', loss.detach(), self.step)
                self.step += 1
                
                if self.step % self.model_save_freq == 0:
                    self.save_model()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.step % self.eval_freq == 0:
                    self.evaluate()
                
                
if __name__ == "__main__":
    # model_path = 'models/model_5.pt'
    model_path = None
    nizami = Nizami(model_path)
    nizami.step = 0
    nizami.train(5000)