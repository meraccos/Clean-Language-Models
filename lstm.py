import torch
import torch.nn.functional as F

from networks import LSTM_Network
from utils import BaseLanguageModel


class LSTM(BaseLanguageModel):
    def __init__(self, model_path = None):
        self.super().__init__()
        self.model_name = "lstm"
        self.n_hidden = 64
        self.batch_size = 128
        self.block_size = 200
        self.num_layers = 1
        self.lr = 0.01
        
        self.emb_size = 8
        
        self.writer = None
        self.prepare_model(model_path)
        
    def prepare_model(self, model_path=None):
        self.model = LSTM_Network(self.n_vocab, self.n_hidden, 
                          self.emb_size, self.num_layers)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
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

    def eval_single_batch(self, x, y):
        output, (h, c) = self.model(x, None)
        output = output.view(-1, self.n_vocab)
        y = y.view(-1)
        loss = F.cross_entropy(output, y)
        return output, loss
    
                
if __name__ == "__main__":
    # model_path = 'models/model_5.pt'
    model_path = None
    lstm = LSTM(model_path)
    lstm.step = 0
    lstm.train(5000)