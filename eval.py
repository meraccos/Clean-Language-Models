from bigram import Bigram
from lstm import LSTM
import argparse
import os
import errno

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation script for the language models")
    parser.add_argument("--model_name", type=str, default="bigram", 
                        help="The model name (eg. 'lstm', 'bigram', etc.)")
    parser.add_argument("--model_idx", type=int, default=None, 
                        help="The index of the saved model to evaluate. \
                            If not provided, the latest model is used.")
    parser.add_argument("--num_chars", type=int, default=100, 
                        help="The number of characters to be generated \
                            for evaluation.")
    args = parser.parse_args()
    
    model_name = args.model_name
    model_idx = args.model_idx
    num_chars = args.num_chars
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    if model_idx is None:
        # Pick the model with the highest index
        files = os.listdir(os.path.join(cwd, 'models', model_name))
        indices = [int(os.path.splitext(f)[0].split('_')[1]) for f in files]
        model_idx = max(indices)
    
    model_idx = str(model_idx)
    
    model_rel_path = f"{model_name}_{model_idx}.pt"
    model_path = os.path.join(cwd, 'models', model_name, model_rel_path)
    isdir = os.path.isfile(model_path)
    if not isdir:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)
    
    name_to_model = {"bigram": Bigram, "lstm": LSTM}
    model = name_to_model[model_name](model_path=model_path)      
    model_out = model.generate(num_chars)
    print(model_out)
    
if __name__ == "__main__":
    main()
