import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

REF_SENTS = [
    "Tom has a red marble, a green marble, a blue marble, and three identical yellow marbles.  How many different groups of two marbles can Tom choose?",
    "The equation x 2 + 2x = i has two complex solutions. Determine the product of their real parts.",
]

# Mean pooling works muuuch better for our use case.
def pooling(outputs: torch.Tensor, inputs: dict) -> np.ndarray:
    return torch.sum(
        outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])

def get_embeddings(model, sents):
    inputs = tokenizer(sents, padding=True, return_tensors='pt')
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    outputs = model(**inputs).last_hidden_state
    embeddings = pooling(outputs, inputs)
    return embeddings

def get_scores(ref_embs, embs):
    return torch.mean(torch.matmul(ref_embs, embs.t()), axis=0).detach().cpu().float().numpy()

if __name__ == '__main__':
    model_name = 'mixedbread-ai/mxbai-embed-large-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name).eval().bfloat16().cuda()
    model = torch.compile(AutoModel.from_pretrained(model_name).eval().bfloat16().cuda())
    ref_embeddings = get_embeddings(model, REF_SENTS)
    test_sents = [
        "I like apples",
        "DNA strands",
        "Joe Biden is President of the USA",
        "The length of a rectangle is 3x + 10 feet and its width is x + 12 feet. If the perimeter of the rectangle is 76 feet, how many square feet are in the area of the rectangle?",
    ]
    embs = get_embeddings(model, test_sents)
    print(get_scores(ref_embeddings, embs))
