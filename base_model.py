import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import get_writer, CharDataset


class BaseLanguageModel:
    """
    Base class for all the language models
    """
    def __init__(
        self,
        dataset="nizami.txt",
        model_name="base",
        batch_size=64,
        block_size=200,
        eval_freq=200,
        save_freq=5000,
        log_freq=500,
        train_portion=0.98
    ):
        self.dataset = dataset
        self.model_name = model_name
        self.batch_size = batch_size
        self.block_size = block_size
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.train_portion = train_portion

        self.step = 0
        self.writer = None

        self.process_data()

    def process_data(self):
        # Process the text file
        with open(os.path.join('datasets', self.dataset), "r") as f:
            text = f.read()

        self.chars = list(text)
        
        # Generate the vocabulary
        v = sorted(list(set(self.chars)))
        self.n_vocab = len(v)

        # Save the encodings for further decoding
        with open(f'encodings/enc_{self.dataset}', 'w') as f:
            f.write(''.join(v))

        # Translation functions
        stoi = {ch: i for i, ch in enumerate(v)}
        itos = {i: ch for i, ch in enumerate(v)}
        
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: "".join(itos[i] for i in l)

        # Split the data
        data = torch.tensor(self.encode(text)).long()

        n = int(self.train_portion * len(data))
        train_ds = CharDataset(data[:n], self.block_size)
        valid_ds = CharDataset(data[n:], self.block_size)

        self.train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.valid_dl = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=True)

    def save_model(self):
        save_path = os.path.join("models", self.model_name)
        save_idx = self.step // self.save_freq
        os.makedirs(save_path, exist_ok=True)
        file_name = os.path.join(save_path, f"{self.model_name}_{save_idx}.pt")
        torch.save(self.model.state_dict(), file_name)

    def eval_single_batch(self, x, y):
        raise NotImplementedError("eval_single_batch not implemented")

    @torch.no_grad()
    def generate(self, n_chars):
        self.model.eval()
        x = torch.tensor(self.encode("\n")[0])
        hidden = None

        idx = [x.item()]
        for _ in range(n_chars):
            x, hidden = self.generate_next_idx(x, hidden)
            idx.append(x.item())

        return self.decode(idx)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        loss_ = 0.0
        for sequence in self.valid_dl:
            output, loss = self.eval_single_batch(sequence)
            loss_ += loss.item()
        
        self.writer.add_scalar("loss/valid", loss_ / len(self.valid_dl), self.step)
        self.writer.add_text("sample", self.generate(100), self.step)
        

    def train(self, max_epoch):
        self.model.train()
        self.writer = self.writer or get_writer(self.model_name, self.dataset)
        loss_ = 0.0

        for epoch in tqdm(range(max_epoch)):
            for sequence in tqdm(self.train_dl, leave=False):
                output, loss = self.eval_single_batch(sequence)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.step += 1
                loss_ += loss.item()
                
                if self.step % self.log_freq == 0:
                    self.writer.add_scalar("loss/train", loss_ / self.log_freq, self.step)
                    loss_ = 0.0

                    if self.step % self.save_freq == 0:
                        self.save_model()

                    if self.step % self.eval_freq == 0:
                        self.evaluate()
                