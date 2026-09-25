import torch
from model import GPTLanguageModel, device, block_size, batch_size, max_iters, learning_rate, eval_iters

with open('pride_and_prejudice.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8 * len(data))
train_data = data[:n]
test_data = data[n:]


def get_batch(split):
    data_split = train_data if split == 'train' else test_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    offsets = torch.arange(block_size)
    x = data_split[ix.unsqueeze(1) + offsets]
    y = data_split[ix.unsqueeze(1) + offsets + 1]
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    model = GPTLanguageModel(vocab_size)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

    best_test_loss = float('inf')

    for iter in range(max_iters):
        if iter % eval_iters == 0:
            losses = estimate_loss(model)
            print(f'step: {iter}, train loss: {losses["train"]:.3f}, test loss: {losses["test"]:.3f}')

            if losses['test'] < best_test_loss:
                best_test_loss = losses['test']
                torch.save(model.state_dict(), 'best_model.pt')
                print(f'  --> Saved new best model checkpoint (test_loss: {best_test_loss:.3f})')

        xb, yb = get_batch('train')
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    print(loss.item())


if __name__ == "__main__":
    main()