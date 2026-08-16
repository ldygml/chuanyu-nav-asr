import random
random.seed(456)

with open('/root/asr-competition/train_data/data.list') as f:
    lines = f.readlines()

indices = list(range(len(lines)))
random.shuffle(indices)

dev_idx = set(indices[:500])
train_idx = set(indices[500:])

with open('/root/asr-competition/train_data/data_train.list', 'w') as f:
    for i in sorted(train_idx):
        f.write(lines[i])
with open('/root/asr-competition/train_data/data_dev.list', 'w') as f:
    for i in sorted(dev_idx):
        f.write(lines[i])

print(f'Train: {len(train_idx)} | Internal Dev: {len(dev_idx)} | External Test: 200 (untouched)')
