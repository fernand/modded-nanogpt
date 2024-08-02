import time

from datasets import load_dataset
from vllm import LLM, SamplingParams

PROMPT = """Given the homework question and answer below, Edit it to minimize the amount of un-necessary knowledge required to actually reason through and answer the question. For example, if the question mentions "bricks" but we don't need to know what a brick is, or mentions a person's name, that's unneccessary knowledge. You must NOT change the sentence structure, but instead replace specific words with more generic words. For example "John" can be replaced with "person A", or "brick" or "door" can be replaced with "object 1" or "object 2". Again, don't use any words or concepts which are not necessary for answering the task. Only answer with the edited text, DO NOT add any other text or explanations. Be sure to edit the answer as well."""

if __name__ == '__main__':
    llm = LLM(
        model='meta-llama/Meta-Llama-3-8B-Instruct',
        enable_prefix_caching=True
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    )
    ds = load_dataset('meta-math/MetaMathQA', split='train')
    print('DS length', len(ds))
    i = 0
    for batch in ds.iter(batch_size=16):
        if i == 2:
            exit()
        seqs = ['\n'.join([PROMPT, '```', batch['query'][i], batch['response'][i], '```']) for i in range(len(batch['query']))]
        conversations = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': seq} for seq in seqs],
            tokenize=False,
        )
        t1 = time.perf_counter()
        outputs = llm.generate(seqs, sampling_params)
        t2 = time.perf_counter()
        print(t2 - t1)
        if i == 0:
            print()
            print(outputs[0].outputs[0].text)
            print()
        i += 1

