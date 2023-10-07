import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


class BaseLanguageModel:
    def __init__(
        self,
        dataset="nizami.txt",
        model_name="base",
        batch_size=6,
        eval_freq=200,
        save_freq=5000,
        block_size=0,
    ):
        self.dataset = dataset
        self.model_name = model_name
        self.batch_size = batch_size
        self.block_size = block_size
        self.eval_freq = eval_freq
        self.save_freq = save_freq

        self.step = 0
        self.writer = None

        self.process_data()

    def process_data(self):
        with open(os.path.join('datasets', self.dataset), "r") as f:
            text = f.read()

        self.chars = list(text)
        v = sorted(list(set(self.chars)))
        self.n_vocab = len(v)
        with open(f'encodings/enc_{self.dataset}', 'w') as f:
            f.write(''.join(v))

        stoi = {ch: i for i, ch in enumerate(v)}
        itos = {i: ch for i, ch in enumerate(v)}
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: "".join(itos[i] for i in l)

        self.data = torch.tensor(self.encode(text), dtype=torch.long)

        n = int(0.95 * len(self.data))
        train_data = self.data[:n]
        val_data = self.data[n:]

        tr_dataset = CharDataset(train_data, self.block_size)
        val_dataset = CharDataset(val_data, self.block_size)

        self.train_dl = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dl = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

    def save_model(self):
        save_path = os.path.join("models", self.model_name)
        save_idx = self.step // self.save_freq
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{self.model_name}_{save_idx}.pt")
        torch.save(self.model.state_dict(), save_file)

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

        self.model.train()
        return self.decode(idx)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        losses = []
        for x, y in self.val_dl:
            output, loss = self.eval_single_batch(x, y)
            losses.append(loss.sum())

        self.writer.add_scalar("Eval/loss", sum(losses) / len(losses), self.step)
        self.model.train()

    def train(self, max_epoch):
        # init the writer here to avoid redundant logging in a new instance
        if self.writer is None:
            self.writer = SummaryWriter(comment="_" + self.model_name)

        for epoch in tqdm(range(max_epoch)):
            for x, y in tqdm(self.train_dl, leave=False):
                output, loss = self.eval_single_batch(x, y)

                self.writer.add_scalar("Train/loss", loss, self.step)
                self.step += 1

                if self.step % self.save_freq == 0:
                    self.save_model()

                if self.step % self.eval_freq == 0:
                    self.evaluate()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


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
