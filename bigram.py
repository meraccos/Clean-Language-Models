import os
import glob
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# import torch.utils.data as data

from model import Bigram_Network


class Bigram:
    def __init__(self, model_path = None):
        self.n_hidden = 128
        self.batch_size = 64
        self.lr = 1e-4
        
        self.step = 0
        self.eval_freq = 200
        self.eval_batches = 20
        self.model_save_freq = 5000
        self.process_books()
        self.prepare_model(model_path)
        self.writer = None
        
    def prepare_model(self, model_path=None):
        self.model = Bigram_Network(self.n_vocab, self.n_hidden)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    
    def process_books(self):  
        with open('books/nizami.txt', 'r') as f:
            text = f.read()

        self.chars = list(text)
        v = sorted(list(set(self.chars)))
        self.n_vocab = len(v)

        stoi = {ch:i for i, ch in enumerate(v)}
        itos = {i:ch for i, ch in enumerate(v)}
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: ''.join(itos[i] for i in l)
        
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - 1, (self.batch_size,))
        x = data[ix]
        y = data[ix + 1]
        return x, y
    
    def save_model(self):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        file_name = f'models/model_{self.step // self.model_save_freq}.pt'
        torch.save(self.model.state_dict(), file_name)
    
    @torch.no_grad()
    def generate(self, n_chars):
        idx = [0]
        x = torch.tensor(0)
        for _ in range(n_chars):
            x = self.model(x)
            x = F.softmax(x, dim=0)
            
            x = torch.multinomial(x, 1)[0]
            idx.append(x.item())
        return self.decode(idx)
    
    @torch.no_grad()
    def evaluate(self):
        losses = 0.0
        for _ in range(self.eval_batches):
            x, y = self.get_batch('test')
            
            output = self.model(x)
            output = output.view(-1, self.n_vocab)
            y = y.view(-1)
            losses += F.cross_entropy(output, y).detach()
        
        mean_loss = losses / self.eval_batches
        self.writer.add_scalar('Eval/loss', mean_loss, self.step)
        
    def train(self, max_epoch):    
        if self.writer is None:
            self.writer = SummaryWriter()    
            
        for epoch in tqdm(range(max_epoch)):
            for _ in range(len(self.chars) // self.batch_size):
                x, y = self.get_batch('train')
                
                output = self.model(x)
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
    model_path = None
    nizami = Bigram(model_path)
    nizami.step = 0
    nizami.train(10)