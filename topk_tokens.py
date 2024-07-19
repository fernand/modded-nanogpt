import glob

import numpy as np
import tiktoken

from train_gpt2 import _load_data_shard

NUM_TOKENS = 50257

if __name__ == '__main__':
    shards = glob.glob('data/fineweb_edu_10B/fineweb_train_*.bin')
    num_tokens = np.zeros((NUM_TOKENS,), dtype=np.int64)
    for shard in shards:
        tokens = _load_data_shard(shard)
        counts = np.bincount(tokens.astype(np.int64), minlength=NUM_TOKENS)
        num_tokens += counts

    enc = tiktoken.get_encoding("gpt2")
    k = int(NUM_TOKENS * 0.1)
    topk = np.argsort(num_tokens)[::-1][:k]
    np.save('topk.npy', topk.astype(np.uint32))
    topk_tokens = [enc.decode([t]) for t in topk]
    with open('topk.txt', 'w') as f:
        f.write('\n'.join(topk_tokens))
