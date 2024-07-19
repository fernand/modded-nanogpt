import argparse
import math

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

REF_SENTS = [
    "Tom has a red marble, a green marble, a blue marble, and three identical yellow marbles.  How many different groups of two marbles can Tom choose?",
    "The equation x 2 + 2x = i has two complex solutions. Determine the product of their real parts.",
]

# Mean pooling works muuuch better for our use case.
def pooling(outputs: torch.Tensor, inputs: dict) -> np.ndarray:
    return torch.sum(
        outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])

def get_embeddings(model, tokenizer, sents):
    inputs = tokenizer(sents, padding=True, return_tensors='pt')
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    outputs = model(**inputs).last_hidden_state
    embeddings = pooling(outputs, inputs)
    return embeddings

def get_scores(ref_embs, embs):
    return torch.mean(torch.matmul(ref_embs, embs.t()), axis=0).detach().cpu().float().numpy()

def get_uuid(id):
    return id.split(':')[-1][:-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate MATH similarity scores')
    parser.add_argument('--num_chunks', type=int, default=2, help='Number of chunks to split the dataset with')
    parser.add_argument('--chunk_idx', type=int, default=0, help='Which chunk index to process, and the GPU id')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.chunk_idx}')
    model_name = 'mixedbread-ai/mxbai-embed-large-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().bfloat16().to(device)
    ref_embeddings = get_embeddings(model, tokenizer, REF_SENTS)

    # test_sents = [
    #     "I like apples",
    #     "DNA strands",
    #     "Joe Biden is President of the USA",
    #     "The length of a rectangle is 3x + 10 feet and its width is x + 12 feet. If the perimeter of the rectangle is 76 feet, how many square feet are in the area of the rectangle?",
    # ]
    # embs = get_embeddings(model, tokenizer, test_sents)
    # print(get_scores(ref_embeddings, embs))

    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train')
    chunk_size = math.ceil(len(ds) / args.num_chunks)
    start_idx = args.chunk_idx * chunk_size
    end_idx = min((args.chunk_idx + 1) * chunk_size, len(ds))
    chunk = ds.select(range(start_idx, end_idx))
    print(f'{len(chunk):_} docs to process start={start_idx}, end={end_idx}')

    fname = f'math_scores_{args.chunk_idx}.csv'
    f = open(fname, 'w')
    for rows in tqdm(chunk.iter(batch_size=32)):
        sents = [doc[:512] for doc in rows['text']]
        embs = get_embeddings(model, tokenizer, sents)
        scores = get_scores(ref_embeddings, embs)
        uuids = [get_uuid(id) for id in rows['id']]
        for uuid, score in zip(uuids, scores):
            f.write(f'{uuid},{score}\n')
    f.close()
