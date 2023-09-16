from train import Nizami
import torch

bigram = Nizami(model_path='models/model_2.pt')
print(bigram.generate(1000))
