import os
import torch

with open("names.txt", "r+") as file:
    contents = file.read()

split_lines = contents.splitlines()
char_set = sorted(set(''.join(split_lines)))

contexts = ['a', 'n', 'n']
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), requires_grad=True, generator=g)
W1 = torch.randn((6, 100), requires_grad=True)
B1 = torch.randn((100,), requires_grad=True)
W2 = torch.randn((100, 27), requires_grad=True)
B2 = torch.randn((27,), requires_grad=True)

parameters = list([C, W1, B1, W2, B2])

block_size = 3

ch2i = {}
i2ch = {}
ch2i['.'] = 0
i2ch[0] = '.'

for i, v in enumerate(char_set):
    ch2i[v] = i + 1
    i2ch[i + 1] = v

combine = []
for v in contexts:
    index = ch2i[v]
    lines = C[index]
    combine += lines

converted = []
X = []
Y = []
for v in split_lines:
    added = list(v)
    context = ['.'] * block_size
    for target in added + ['.']:
        nums = []
        for c in context:
            nums.append(ch2i[c])
        X.append(nums)
        Y.append(ch2i[target])
        converted.append((context, target))
        context = context[1:] + [target]

tx = torch.tensor(X)
ty = torch.tensor(Y)

# print(converted[:10])
# print(tx)
# print(ty)
# print(tx.shape)

# print(len(split_lines))
# total = 0
# for i, v in enumerate(split_lines):
#     total += len(v)
# print( total + len(split_lines))
# print(C[5].shape)
# print(C[[5, 13, 13]], C[[5, 13, 13]].shape)

torch.cat([C[5], C[13], C[13]])

dim_N_6 = []
# for x in tx[3:4]:
#     tmp = []
#     for i in x:
#         tmp.append(C[i])
#     conc = torch.cat(tmp)
#     dim_N_6.append(conc)

#最终写法，当前最优解
# dim_N_6 = C[tx].view(-1, 6)

for _ in range(10000):
    batch_indices = torch.randint(0, len(tx), (32,))
    batch_x = tx[batch_indices]
    batch_y = ty[batch_indices]

    emb = C[batch_x]
    dim_N_6 = torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], dim=1)
    hidden = torch.tanh(dim_N_6 @ W1 + B1)
    logits = hidden @ W2 + B2
    F = torch.nn.functional
    loss = F.cross_entropy(logits, batch_y)
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data -= 0.1 * p.grad

loss
# a = torch.tensor([-100, -3, 0, 100]).float()
# b = torch.tensor(2)
# result = a.exp() / a.exp().sum(0, keepdim=True)
# print(result)
# print(F.cross_entropy(a, b))
