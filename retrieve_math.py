from typing import Dict

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# The model works really well with cls pooling (default) but also with mean pooling.
def pooling(outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()

model_id = 'mixedbread-ai/mxbai-embed-large-v1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).cuda()

docs = [
    "Tom has a red marble, a green marble, a blue marble, and three identical yellow marbles.  How many different groups of two marbles can Tom choose?",
    "The equation x 2 + 2x = i has two complex solutions. Determine the product of their real parts.",
    "I like apples",
    "DNA strands",
    "Joe Biden is President of the USA",
    "The length of a rectangle is 3x + 10 feet and its width is x + 12 feet. If the perimeter of the rectangle is 76 feet, how many square feet are in the area of the rectangle?",
]

inputs = tokenizer(docs, padding=True, return_tensors='pt')
for k, v in inputs.items():
    inputs[k] = v.cuda()
outputs = model(**inputs).last_hidden_state
embeddings = pooling(outputs, inputs, 'mean')

mask = [False, True, True, True, True, True]
print(embeddings[mask] @ embeddings[0])
