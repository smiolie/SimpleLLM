import torch
from model import GPTLanguageModel, device, block_size

# Rebuild vocab (must match what was used at training time)
with open('pride_and_prejudice.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# Load model
model = GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.to(device)
model.eval()

# Interactive prompt loop
while True:
    prompt = input("Prompt (or 'quit'): ")
    if prompt.lower() == 'quit':
        break
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))