from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import CrossEntropyLoss
from torch import tensor
import pandas as pd
device = torch.device('cuda')
import numpy as np
import tqdm

# most of the code taken from https://github.com/potamides/unsupervised-metrics/blob/master/metrics/utils/perplexity.py
def lm_perplexity(hyps, device, name="gpt2"):
    # Some models need a special tokenizer, like chinese gpt2, see here:
    # https://huggingface.co/ckiplab/gpt2-base-chinese
    model_name, tokenizer_name = (name, name) if isinstance(name, str) else name

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    scores = list()
    model.eval()
    for hyp in tqdm.tqdm(hyps):
        tokenize_input = tokenizer.tokenize(hyp.lower())

        if len(tokenize_input) <= 1:
            scores.append(0)
        else:
            if len(tokenize_input) > 1024:
                tokenize_input = tokenize_input[:1024]

            input_ids = tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
            score = model(input_ids, labels=input_ids)
            scores.append(np.exp(score.loss.item()))

    return np.average(scores)


def main():
    df = pd.read_csv('../dataset/four_line_poetry.csv')
    print("Perplexity: ", lm_perplexity(df.loc[df.label == 'abba'].poem.values, device))


if __name__ == 'main':
    main()