import os

data_dir = "/root/asr-competition/official_data"
feat_dir = f"{data_dir}/feat"
dict_file = "/root/asr-competition/model_checkpoints/dict.chr7531.txt"

# Load dictionary
char2id = {}
with open(dict_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            char2id[parts[0]] = int(parts[1])
vocab_size = len(char2id)

# Load text
utt2txt = {}
with open(f"{data_dir}/text") as f:
    for line in f:
        parts = line.strip().split('\t', 1)
        if len(parts) >= 2:
            utt2txt[parts[0]] = parts[1]

# Load feat lengths
utt2len = {}
with open(f"{feat_dir}/feat2len.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            utt2len[parts[0]] = int(parts[1])

# Read feat scp and generate data.list
feat_dim = 40
datalist_file = f"{data_dir}/data.list"
matched = 0

with open(f"{feat_dir}/mfcc.scp") as fin, open(datalist_file, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        utt, ark = parts[0], parts[1]

        if utt not in utt2txt or utt not in utt2len:
            continue

        txt = utt2txt[utt]
        feat_len = utt2len[utt]
        token = " ".join(list(txt.replace(" ", "")))
        token_shape = len(token.split())
        token_ids = [str(char2id.get(ch, 1)) for ch in token.split()]
        tokenid_str = ",".join(token_ids)

        res = (f"utt:{utt}\tfeat:{ark}\tfeat_shape:{feat_len},{feat_dim}\t"
               f"text:{txt}\ttoken:{token}\ttokenid:[{tokenid_str}]\t"
               f"token_shape:{token_shape},{vocab_size}")
        fout.write(res + "\n")
        matched += 1

print(f"Official data.list: {matched} lines")

# Now merge with existing training data
train_file = "/root/asr-competition/train_data/data_train.list"
merged = []
with open(train_file) as f:
    for line in f:
        merged.append(line.strip())
with open(datalist_file) as f:
    for line in f:
        merged.append(line.strip())

# Shuffle merged data
import random
random.seed(789)
random.shuffle(merged)

# New split: 500 dev, 200 test (keep original valid_data as external test)
# Actually keep original external 200 test untouched, create new internal split
dev_size = 500

dev_data = merged[:dev_size]
train_data = merged[dev_size:]

with open("/root/asr-competition/train_data/data_train_v2.list", "w") as f:
    for l in train_data:
        f.write(l + "\n")
with open("/root/asr-competition/train_data/data_dev_v2.list", "w") as f:
    for l in dev_data:
        f.write(l + "\n")

print(f"Merged train: {len(train_data)} lines")
print(f"Merged dev: {len(dev_data)} lines")
print(f"Total: {len(merged)} lines (35093 + {matched})")
