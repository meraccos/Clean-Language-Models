import os
import glob
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import LSTM


class Nizami:
    def __init__(self, model_path = None):
        self.n_hidden = 128
        self.batch_size = 128
        self.seq_length = 32
        self.num_layers = 4
        self.lr = 0.001
        
        self.process_books()
        self.prepare_model(model_path)
        self.writer = SummaryWriter()
        
    def prepare_model(self, model_path=None):
        self.model = LSTM(self.n_vocab, self.n_hidden, 
                          self.n_vocab, self.num_layers)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def process_books(self):
        files = glob.glob('books/*.txt')
        all_book_lines = []   

        for filename in files:
            with open(filename, 'r') as f:
                all_book_lines += f.readlines()

        self.chars = [char for line in all_book_lines for char in line]
        self.v = list(set(self.chars))
        self.n_vocab = len(self.v)
        self.v_idx = {self.v[i]:i for i in range(self.n_vocab)} 
        
    def str_to_idx(self, word):
        idxs = torch.tensor([self.v_idx[letter] for letter in word])
        return idxs
        
    def str_to_oh(self, word):
        idxs = self.str_to_idx(word)
        oh = F.one_hot(idxs, self.n_vocab)
        return oh
    
    def get_training_batch(self, data):
        X_batch = torch.zeros(self.seq_length, self.batch_size, self.n_vocab).float()
        Y_batch = torch.zeros(self.seq_length, self.batch_size, dtype=torch.long)
        
        for i in range(self.batch_size):
            start_idx = random.randint(0, len(data) - self.seq_length - 1)
            x_list = data[start_idx:start_idx + self.seq_length]
            y_list = data[start_idx + 1:start_idx + self.seq_length + 1]

            X_oh = self.str_to_oh(x_list).type(torch.float32)
            Y_idx = self.str_to_idx(y_list)

            X_batch[:, i] = X_oh
            Y_batch[:, i] = Y_idx

        return X_batch, Y_batch
    
    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        file_name = f'models/model_{epoch // 100}.pt'
        torch.save(self.model.state_dict(), file_name)
    
    def train(self, max_epoch):
        for epoch in tqdm(range(max_epoch)):
            X, y = self.get_training_batch(self.chars)
            
            h, c = self.model.init_hidden(self.batch_size)
            output, (h, c) = self.model(X, (h.detach(), c.detach()))

            output = output.view(-1, self.n_vocab)
            y = y.view(-1)
            loss = self.criteria(output, y)
            
            self.writer.add_scalar('Loss/train', loss.detach(), epoch)
            
            if epoch % 500 == 0:
                self.save_model(epoch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
if __name__ == "__main__":
    nizami = Nizami()
    nizami.train(3000)