from bigram import Bigram
import torch

bigram = Bigram(model_path='models/model_40.pt')
print(bigram.generate(1000))
