import subprocess, sys, json, os, glob

ckpt = sys.argv[1]  # model checkpoint path
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(ckpt).replace('.pt','')
data_dir = '/root/asr-competition/valid_data'
work_dir = f'/tmp/eval_{label}'
os.makedirs(work_dir, exist_ok=True)

# Step 1: Inference
print(f'[1/4] Running inference on {label}...')
result = subprocess.run([
    'python', 'infer.py',
    '--config-dir', 'config', '--config-name', 'infer',
    'task=spec_finetuning',
    f'task.data={data_dir}',
    'task.normalize=false',
    'common.user_dir=/root/asr-competition/ASR/data2vec_dialect',
    f'common_eval.path={ckpt}',
    f'common_eval.results_path={work_dir}',
    'common_eval.quiet=true',
    'dataset.gen_subset=dev',
    '+task.target_dictionary=/root/asr-competition/model_checkpoints'
], capture_output=True, cwd='/root/asr-competition/ASR/data2vec_dialect')

# Step 2: Extract HYPO/REF to JSONL
print(f'[2/4] Extracting predictions...')
log = glob.glob(f'{work_dir}/dev/infer.log')[0]
hypos, refs = [], []
with open(log) as f:
    for l in f:
        if 'HYPO:' in l: hypos.append(l.split('HYPO: ',1)[1].strip())
        elif 'REF:' in l: refs.append(l.split('REF: ',1)[1].strip())

pred_path = f'{work_dir}/pred.jsonl'
ref_path = f'{work_dir}/ref.jsonl'
with open(pred_path,'w') as f:
    for i,h in enumerate(hypos): f.write(json.dumps({'id':f's{i:04d}','text':h},ensure_ascii=False)+'\n')
with open(ref_path,'w') as f:
    for i,r in enumerate(refs): f.write(json.dumps({'id':f's{i:04d}','text':r},ensure_ascii=False)+'\n')

# Step 3: Normalize
print(f'[3/4] Normalizing...')
subprocess.run(['python3','normalize.py','-m','jsonl','-i',pred_path,'-o',f'{work_dir}/pred_norm.jsonl'], cwd='/root/asr-competition/ASR/data2vec_dialect')
subprocess.run(['python3','normalize.py','-m','jsonl','-i',ref_path,'-o',f'{work_dir}/ref_norm.jsonl'], cwd='/root/asr-competition/ASR/data2vec_dialect')

# Step 4: Score
print(f'[4/4] Computing CER...')
r = subprocess.run(['python3','score.py','-r',f'{work_dir}/ref_norm.jsonl','-t',f'{work_dir}/pred_norm.jsonl','-o',f'{work_dir}/cer.jsonl'], capture_output=True, text=True, cwd='/root/asr-competition/ASR/data2vec_dialect')
for line in r.stdout.split('\n'):
    if '平均 CER' in line:
        print(f'\n=== {label} ===\n{line}')
