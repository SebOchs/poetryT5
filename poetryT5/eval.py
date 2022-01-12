from transformers import T5ForConditionalGeneration, AutoTokenizer, Adafactor
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch
import time

model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

# To decoder only
del model.encoder
enc_out = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=torch.zeros([1, 5, 1472]))
o = Adafactor(model.parameters(), lr=None, warmup_init=True, relative_step=True, scale_parameter=True)

def train(iters=1):
    start = time.time()
    input_ids = tokenizer([
            "aabb",
            "aabb",
            "abab",
        ]
        , return_tensors="pt", padding='longest').input_ids
    aabb = torch.tensor([100, 100, 101, 101,   1])
    abab = torch.tensor([100, 101, 100, 101,   1])
    abba = torch.tensor([100, 101, 101, 100,   1])
    hidden_state = torch.zeros(input_ids.shape[0], 5, 1472)
    for i, id in enumerate(input_ids):
        if torch.all(id == aabb):
            hidden_state[i] = 0
        elif torch.all(id == abab):
            hidden_state[i] = 0.1
        elif torch.all(id == abba):
            hidden_state[i] = 0.2
    encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_state)
    labels = tokenizer([
            "Sweet surrender of finding\nOne's tired soul unwinding\nUntil life no longer matters;\nLove unchains all its fetters.",
            "'Letters to the monarch tell\n\"How Alhama's city fell:\"\nIn the fire the scroll he threw,\nAnd the messenger he slew.",
            "Not for to hide it in a hedge,\nNor for a train attendant;\nBut for the glorious privilege\nOf being independent.",
        ], return_tensors="pt", padding='longest').input_ids
    for i in range(iters):
        o.zero_grad()
        outputs = model(encoder_outputs=encoder_outputs, labels=labels)
        outputs.loss.backward()
        o.step()
    print(f'Elapsed time: {time.time() - start}')

print('Start Training')
train(500)
print('End Training 100')

def inference(input_string="aabb"):
    hidden_state = torch.zeros(1, 5, 1472)
    if input_string=="aabb":
        hidden_state[0] = 0
    elif input_string=="abab":
        hidden_state[0] = 0.1
    elif input_string=="abba":
        hidden_state[0] = 0.2
    encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_state)
    generation = model.generate(encoder_outputs=encoder_outputs, max_length=200, do_sample=True, num_return_sequences=3)
    for x in generation:
        print(tokenizer.decode(x, skip_special_tokens=True))
        print('---------')

inference()