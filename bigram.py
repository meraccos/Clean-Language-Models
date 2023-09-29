import torch
import torch.nn.functional as F

from networks import Bigram_Network
from utils import BaseLanguageModel


class Bigram(BaseLanguageModel):
    def __init__(self, model_path=None):
        super().__init__(
            dataset="tinyshakespeare.txt",
            model_name="bigram",
            batch_size=64,
            eval_freq=200,
            save_freq=5000,
            block_size=1,
        )
        self.n_hidden = 128
        self.lr = 1e-4
        self.prepare_model(model_path)

    def prepare_model(self, model_path=None):
        self.model = Bigram_Network(self.n_vocab, self.n_hidden)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def generate_next_idx(self, x, hidden):
        x = self.model(x)
        x = F.softmax(x, dim=0)
        x = torch.multinomial(x, 1)[0]
        return x, hidden

    def eval_single_batch(self, x, y):
        output = self.model(x)
        output = output.view(-1, self.n_vocab)
        y = y.view(-1)
        loss = F.cross_entropy(output, y)
        return output, loss


if __name__ == "__main__":
    model_path = None
    nizami = Bigram(model_path)
    nizami.step = 0
    nizami.train(10)
