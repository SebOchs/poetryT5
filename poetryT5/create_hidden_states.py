from transformers import T5ForConditionalGeneration, AutoTokenizer
import numpy as np
import os

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

# Calculate encoder output for pretrained model
input = tokenizer(['aabb', 'abab', 'abba'], return_tensors="pt")
out = model.encoder(input_ids=input['input_ids'], attention_mask=input['attention_mask'])['last_hidden_state']
out = out.detach().cpu().numpy()

# Save hidden states
if not os.path.exists('dataset'):
    os.makedirs('dataset')
np.save('dataset/encoder_hidden.npy', {
    'aabb_small': out[0],
    'abab_small': out[1],
    'abba_small': out[2],
})