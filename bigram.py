from pathlib import Path

import torch


# 读取与脚本放在同一目录下的姓名数据。
with Path(__file__).with_name("names.txt").open(encoding="utf-8") as file:
    contents = file.read()

split_line_list = contents.splitlines()

# 给每个姓名添加起始和结束标记。
handled = []
for word in split_line_list:
    word_split = list(word)
    word_split.insert(0, "<S>")
    word_split.append("<E>")
    handled.append(word_split)

# 统计相邻字符对出现的次数。
b = {}
for chs in handled:
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

print("不同字符对数量：", len(b))
print("出现最多的 10 个字符对：", sorted(b.items(), key=lambda kv: -kv[1])[:10])

# 建立双向映射，用索引 0 对应的 '.' 统一表示姓名边界。
chars = sorted(set("".join(split_line_list)))
char_to_index = {".": 0}
index_to_char = {}

for i, ch in enumerate(chars):
    char_to_index[ch] = i + 1

for ch, i in char_to_index.items():
    index_to_char[i] = ch

# N[x, y] 是字符 x 后面出现字符 y 的次数。
N = torch.zeros((27, 27), dtype=torch.int32)
for pair, num in b.items():
    start, end = pair
    t_start = "." if start in ("<S>", "<E>") else start
    t_end = "." if end in ("<S>", "<E>") else end
    x = char_to_index[t_start]
    y = char_to_index[t_end]
    N[x, y] = num

# 加一平滑后按行归一化，得到下一个字符的条件概率。
P = (N + 1) / (N + 1).sum(1, keepdim=True)


def generate_name(gen):
    """生成姓名，并返回姓名及各字母的转移概率（不含结束标记）。"""
    prob = []
    final_char = ""
    ix = 0

    while True:
        p = P[ix]
        index = torch.multinomial(p, num_samples=1, replacement=True, generator=gen)
        ix = index.item()
        if ix == 0:
            break
        final_char += index_to_char[ix]
        prob.append(p[index])

    return final_char, prob


def calculate_loss(name, P, ch2i):
    """返回含起始、结束转移的负对数似然总和，以及字符对数量。"""
    pair_num = len(name) + 1
    log_sum = 0.0
    cons_name = ["."] + list(name) + ["."]
    pairs = zip(cons_name, cons_name[1:])
    probs = []

    for s, e in pairs:
        x = ch2i[s]
        y = ch2i[e]
        probs.append(P[x, y])

    for prob in probs:
        log_sum += -torch.log(prob).item()

    return log_sum, pair_num


# 固定随机种子，便于复现采样结果。
g = torch.Generator().manual_seed(2147483647)
r = generate_name(g)
print("生成的姓名：", r[0])

# 按字符对数量汇总 loss，而不是直接平均每个姓名的 loss。
names = ["andrejq", "emma", "tom"]
total_log_sum = 0.0
total_pair_num = 0

for name in names:
    log_sum, pair_num = calculate_loss(name, P, char_to_index)
    print(f"{name} 的负对数似然总和：{log_sum}")
    total_log_sum += log_sum
    total_pair_num += pair_num

print("平均负对数似然：", total_log_sum / total_pair_num)
