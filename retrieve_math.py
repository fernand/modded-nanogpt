import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

input_texts = [
    "Tom has a red marble, a green marble, a blue marble, and three identical yellow marbles.  How many different groups of two marbles can Tom choose?",
    "The equation x 2 + 2x = i has two complex solutions. Determine the product of their real parts.",
    "I like apples",
    "DNA strands",
    "Joe Biden is President of the USA",
    "The length of a rectangle is 3x + 10 feet and its width is x + 12 feet. If the perimeter of the rectangle is 76 feet, how many square feet are in the area of the rectangle?",
]

model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()

batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
batch_dict = {key: value.cuda() for key, value in batch_dict.items()}

outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state[:, 0]

embeddings = F.normalize(embeddings, p=2, dim=1)
print((embeddings[1:] @ embeddings[0]).tolist())
# mask = [True, False, True, True, True]
# print((embeddings[mask] @ embeddings[1]).tolist())