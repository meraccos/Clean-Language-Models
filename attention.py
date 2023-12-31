import torch
import torch.nn.functional as F

from base_model import BaseLanguageModel
from networks import Attention_Network
from networks import Encoder


class Attention(BaseLanguageModel):
    def __init__(self, model_path=None):
        super().__init__(
            dataset="nizami.txt",
            model_name="attention",
            batch_size=256,
            eval_freq=200,
            save_freq=500,
            block_size=100,
        )
       
        self.embd = 12
        self.ff_dim = 128
        self.nhead = 3
        self.n_layers = 3
        self.lr = 0.02
        self.prepare_model(model_path)

    def prepare_model(self, model_path=None):
        self.model = Attention_Network(
            self.n_vocab, self.embd, self.nhead, self.n_layers, self.ff_dim, self.block_size
        )
        
        self.model = Encoder(72, 
              self.embd, 
              self.n_layers, 
              self.nhead, 
              self.ff_dim, 
              0.1, 
              'cpu')
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def generate_next_idx(self, x, hidden):
        x = x.view(1)
        x, hidden = self.model(x, hidden)
        x = F.softmax(x, dim=1)
        x = torch.multinomial(x, 1).squeeze(0)
        x = x.type(torch.int)
        return x, hidden

    def eval_single_batch(self, x, y):
        m = torch.nn.Transformer.generate_square_subsequent_mask(self.block_size)
        output = self.model(x, src_mask=m)        
        loss = F.cross_entropy(output.contiguous().view(-1, self.n_vocab), y.contiguous().view(-1))
        
        return output, loss

if __name__ == '__main__':
    attention = Attention()
    print('hi')
    attention.train(2)