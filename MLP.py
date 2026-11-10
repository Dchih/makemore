import os
import torch
import random
import matplotlib.pyplot as plt
F = torch.nn.functional
torch.set_num_threads(1)

with open("names.txt", "r+") as file:
    contents = file.read()

split_lines = contents.splitlines()
random.shuffle(split_lines)
char_set = sorted(set(''.join(split_lines)))

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), requires_grad=True, generator=g)
W1 = torch.randn((30, 200), generator=g) * 0.1
B1 = torch.randn((200,), generator=g) * 0.5
W2 = torch.randn((200, 27), generator=g) * 0.01
B2 = torch.randn((27,), generator=g) * 0
block_size = 3

parameters = list([C, W1, B1, W2, B2])
print(W2.is_leaf, B2.is_leaf)
for p in parameters:
    p.requires_grad = True

ch2i = {}
i2ch = {}
ch2i['.'] = 0
i2ch[0] = '.'

for i, v in enumerate(char_set):
    ch2i[v] = i + 1
    i2ch[i + 1] = v

def convert(names):
    X = []
    Y = []
    for name in names:
        context = ['.'] * block_size
        name_chars = list(name)
        # 保留结束符样本，让名字生成能学习何时停止。
        for ch in name_chars + ['.']:
            nums = []
            for c in context:
                nums.append(ch2i[c])
            X.append(nums)
            Y.append(ch2i[ch])
            context = context[1:] + [ch]
    tx = torch.tensor(X)
    ty = torch.tensor(Y)
    return tx, ty

train_set_size = int(0.8 * len(split_lines))
dev_set_size = int(0.1 * len(split_lines)) + train_set_size
test_set_size = dev_set_size + int(0.1 * len(split_lines))

train_set = split_lines[0:train_set_size]
dev_set = split_lines[train_set_size:dev_set_size]
test_set = split_lines[dev_set_size:test_set_size]

Xtr, Ytr = convert(train_set)
Xdev, Ydev = convert(dev_set)
Xtest, Ytest = convert(test_set)

lossi = []
for _ in range(300000):
    batch = torch.randint(0, len(Xtr), (32,))
    batch_x = Xtr[batch]
    batch_y = Ytr[batch]
    dim = C[batch_x].view(-1, 30)
    preact = dim @ W1 + B1
    hidden = torch.tanh(preact)
    logits = hidden @ W2 + B2
    loss = F.cross_entropy(logits, batch_y)
    lossi.append(loss.item())
    for p in parameters:
        p.grad = None
    loss.backward()
    with torch.no_grad():
        for p in parameters:
            p.data -= 0.1 * p.grad

# 以下激活值和 loss 来自最后一次训练前向，处于最后一次参数更新之前。
print(loss)
print(W1.std())
print(B1.std())
print(preact[:, 0])
print(batch_x[:5])
print(C.std())
print(C[:3])
print(dim[:3, :6])
print(B1[0])
print(Xtr[:8])
print(Ytr[:8])
print(batch[:8])
print((Xtr != 0).sum())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(preact.detach().view(-1).tolist(), bins=50, rwidth=0.9)
axes[0].set_title("Pre-activation (last training batch)")
axes[1].hist(hidden.detach().view(-1).tolist(), bins=50, rwidth=0.9)
axes[1].set_title("tanh output (last training batch)")
for ax in axes:
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
fig.tight_layout()

# 用更新后的参数重新评估，不反向传播，也不更新参数。
with torch.no_grad():
    dim_train = C[Xtr].view(-1, 30)
    hidden_train = torch.tanh(dim_train @ W1 + B1)
    logits_train = hidden_train @ W2 + B2
    loss2 = F.cross_entropy(logits_train, Ytr)
    print(loss2)

    dim_dev = C[Xdev].view(-1, 30)
    preact_dev = dim_dev @ W1 + B1
    hidden_dev = torch.tanh(preact_dev)
    logits_dev = hidden_dev @ W2 + B2
    loss_dev = F.cross_entropy(logits_dev, Ydev)
    print(loss_dev)

def generate_name():
    final_pred = ''
    context = [0, 0 ,0]
    while True:
        ctx_tensor = torch.tensor(context)
        with torch.no_grad():
            dim_N_X = C[ctx_tensor].view(1, -1)
            hidden = torch.tanh(dim_N_X @ W1 + B1)
            logits = hidden @ W2 + B2
            probs = F.softmax(logits, dim=1)
            index = torch.multinomial(probs, num_samples=1, replacement=True, generator=g)
        if index.item() == 0:
            break
        ch = i2ch[index.item()]
        final_pred += ch
        context = context[1:] + [index.item()]
    return final_pred

for _ in range(20):
    print(generate_name())

plt.figure()
plt.plot(lossi)
plt.xlabel("training step")
plt.ylabel("batch loss")
plt.show()
