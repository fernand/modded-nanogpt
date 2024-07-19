import multiprocessing as mp

import tiktoken
from datasets import load_dataset

ds = load_dataset('meta-math/MetaMathQA', split='train')
enc = tiktoken.get_encoding('o200k_base')

def tokenize_to_bytes(doc):
    tokens = enc.encode_ordinary(doc['query'])
    tokens.extend(enc.encode_ordinary(doc['response']))
    return [enc.decode_single_token_bytes(t) for t in tokens]

if __name__ == '__main__':
    all_tokens_bytes = set()
    with mp.Pool(10) as pool:
        for tokens in pool.imap(tokenize_to_bytes, ds, chunksize=16):
            all_tokens_bytes = all_tokens_bytes.union(tokens)
    with open('metamath_qa_tokens.txt', 'wb') as f:
        for encoded in all_tokens_bytes:
            f.write(encoded)
            f.write(b'\n')
