#!/usr/bin/env python3
"""Hotword correction driven by the approved dictionary (词典/hotword_dict_final.md):
   replace EVERY occurrence of each 错->对 pair in the text, subject to the
   substring-damage guard (a lengthening replacement touching a hanzi neighbor
   is skipped at that occurrence). REVERT_SET pairs are skipped entirely.
Usage: python hotword_fix.py <in.jsonl> <out.jsonl>
"""
import sys, json, re, os

# 词典路径：环境变量 HOTWORD_DICT 优先；默认取脚本同级的 词典/hotword_dict_final.md
DICT = os.environ.get('HOTWORD_DICT') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '词典', 'hotword_dict_final.md')

# load pairs 错 -> 对
pairs = []
for l in open(DICT, encoding='utf-8'):
    p = l.strip().split()
    if len(p) >= 2:
        pairs.append((p[0], p[1] if len(p) == 2 else ' '.join(p[1:])))
# longest wrong-string first so nested pairs don't double-apply
pairs_sorted = sorted((p for p in pairs if p[0] and p[1]), key=lambda p: -len(p[0]))

punc_re = re.compile(r'[^一-鿿A-Za-z0-9]')

def is_han(c):
    return '一' <= c <= '鿿'

# manual revert exceptions (user-flagged / dev-confirmed substring damage)
REVERT_SET = {('久曲路', '旧曲路'), ('经点', '景点')}

# context exceptions: pair -> preceding char that marks an occurrence as invalid
# (e.g. 约后 must not fire inside 预约后天)
PREV_EXCEPT = {('约后', '约好'): '预'}

def correct_one(text):
    new = punc_re.sub('', text)
    kept, reverted = [], []
    for w, r in pairs_sorted:
        if (w, r) in REVERT_SET:
            continue
        start = 0
        while True:
            idx = new.find(w, start)
            if idx < 0:
                break
            if (w, r) in PREV_EXCEPT and idx > 0 and new[idx - 1] == PREV_EXCEPT[(w, r)]:
                reverted.append((w, r))
                start = idx + len(w)
                continue
            left_han = idx > 0 and is_han(new[idx - 1])
            right_han = idx + len(w) < len(new) and is_han(new[idx + len(w)])
            # substring damage guard: lengthening replace touching a hanzi neighbor
            if len(r) > len(w) and (left_han or right_han):
                reverted.append((w, r))
                start = idx + len(w)
                continue
            new = new[:idx] + r + new[idx + len(w):]
            kept.append((w, r))
            start = idx + len(r)
    return new, kept, reverted

def main():
    in_fn, out_fn = sys.argv[1], sys.argv[2]
    preds = [json.loads(l) for l in open(in_fn, encoding='utf-8')]
    out = []
    n_kept = n_reverted = n_changed = 0
    kept_pairs, reverted_pairs = {}, {}
    for item in preds:
        new, kept, reverted = correct_one(item['text'])
        if kept:
            n_changed += 1
        n_kept += len(kept)
        n_reverted += len(reverted)
        for w, r in kept:
            kept_pairs[(w, r)] = kept_pairs.get((w, r), 0) + 1
        for w, r in reverted:
            reverted_pairs[(w, r)] = reverted_pairs.get((w, r), 0) + 1
        out.append({'id': item['id'], 'text': new})

    with open(out_fn, 'w', encoding='utf-8') as f:
        for l in out:
            f.write(json.dumps(l, ensure_ascii=False) + '\n')

    print(f'utts: {len(preds)}, changed: {n_changed}')
    print(f'kept replacements: {n_kept}, reverted: {n_reverted}')
    print(f'\nTop kept ({len(kept_pairs)} kinds):')
    for (w, r), c in sorted(kept_pairs.items(), key=lambda x: -x[1])[:15]:
        print(f'  {c:4d}  {w} -> {r}')
    print(f'\nTop reverted ({len(reverted_pairs)} kinds):')
    for (w, r), c in sorted(reverted_pairs.items(), key=lambda x: -x[1])[:15]:
        print(f'  {c:4d}  {w} -> {r}')
    print(f'\nSaved to {out_fn}')

if __name__ == '__main__':
    main()
