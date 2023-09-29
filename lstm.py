import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from torch.utils.data import DataLoader

from model import LSTM_Network, CharDataset
    

class LSTM:
    def __init__(self, model_path = None):
        self.n_hidden = 64
        self.batch_size = 128
        self.block_size = 200
        self.num_layers = 1
        self.lr = 0.01
        
        self.emb_size = 8
        
        self.step = 0
        self.eval_freq = 200
        self.model_save_freq = 5000
        self.writer = None
        self.process_books()
        self.prepare_model(model_path)
        
    def prepare_model(self, model_path=None):
        self.model = LSTM_Network(self.n_vocab, self.n_hidden, 
                          self.emb_size, self.num_layers)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    
    def process_books(self):  
        with open('books/nizami.txt', 'r') as f:
            text = f.read()

        self.chars = list(text)
        v = list(set(self.chars))
        self.n_vocab = len(v)

        stoi = {ch:i for i, ch in enumerate(v)}
        itos = {i:ch for i, ch in enumerate(v)}
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: ''.join(itos[i] for i in l)
        
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        
        n = int(0.95 * len(self.data))
        train_data = self.data[:n]
        val_data = self.data[n:]
        
        train_dataset = CharDataset(train_data, self.block_size)
        val_dataset = CharDataset(val_data, self.block_size)

        self.train_dl = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dl = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
    
    def save_model(self):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        file_name = f'models/model_{self.step // self.model_save_freq}.pt'
        torch.save(self.model.state_dict(), file_name)
        
    def generate(self, n_gen_chars=150):
        self.model.eval()
        x = torch.tensor(0).view(1)
        ix = [x.item()]

        hidden = None
        for _ in range(n_gen_chars):
            x, hidden = self.model(x, hidden)
            x = F.softmax(x, dim=1)
            x = torch.multinomial(x, 1).squeeze(0)
            x = x.type(torch.int)
            ix.append(x.item())
        self.model.train()
        return self.decode(ix)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        losses = []
        for x, y in self.val_dl:
            output, (h, c) = self.model(x, None)
            output = output.view(-1, self.n_vocab)
            y = y.view(-1)
            loss = F.cross_entropy(output, y).detach().item()
            losses.append(loss)

        self.writer.add_scalar('Eval/loss', sum(losses) / len(losses), self.step)
        self.model.train()

    def train(self, max_epoch):
        # init the writer here to avoid redundant logging during generation
        if self.writer == None:    
            self.writer = SummaryWriter()    
            
        for epoch in range(max_epoch):
            for x, y in tqdm(self.train_dl):
                output, (h, c) = self.model(x, None)
                output = output.view(-1, self.n_vocab)
                y = y.view(-1)
                loss = F.cross_entropy(output, y)
                
                self.writer.add_scalar('Loss/train', loss.detach(), self.step)
                self.step += 1
                
                if self.step % self.model_save_freq == 0:
                    self.save_model()
                
                if self.step % self.eval_freq == 0:
                    self.evaluate()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                
if __name__ == "__main__":
    # model_path = 'models/model_5.pt'
    model_path = None
    lstm = LSTM(model_path)
    lstm.step = 0
    lstm.train(5000)