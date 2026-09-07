import json, random, hashlib, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import peft
SNAP=200
BASE='/mnt/workspace/models/REAL-Prover-fe76f68d'
SRC='/mnt/workspace/new_value_head/dataset/train.jsonl'
OUT='/mnt/workspace/head-runs/datasets/fullv3-train-features'
os.makedirs(OUT, exist_ok=True)
rows=[]
with open(SRC) as f:
    for line in f:
        try: rows.append(json.loads(line))
        except: pass
print('rows', len(rows))
import io, base64
d=json.load(open('/root/exp1/artifact/backend.json'))
payload=torch.load(io.BytesIO(base64.b64decode(d['payload'])), map_location='cpu', weights_only=True)
dev=torch.device('cuda')
tok=AutoTokenizer.from_pretrained(BASE)
model=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={'':dev}, use_cache=False); model.eval()
cfg=peft.LoraConfig(r=16,lora_alpha=32,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
pm=peft.get_peft_model(model,cfg); pm.load_state_dict(payload['adapter'], strict=False); pm.eval()
n=len(rows); done=0
with torch.no_grad():
    for start in range(0, n, SNAP):
        out_path=f'{OUT}/features-{start:06d}-{min(start+SNAP,n):06d}.pt'
        if os.path.exists(out_path): done+=1; continue
        batch=rows[start:start+SNAP]
        enc=tok([r['prompt'] for r in batch], return_tensors='pt', padding=True, truncation=True, max_length=256).to(dev)
        out=pm(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        feats=out.hidden_states[-1][:,-1,:].to(torch.float16).cpu()
        labels=torch.tensor([max(0,min(63,int(r.get('value_class',64)))) for r in batch])
        torch.save({'schema_version':'new_value_head.feature-shard.v2','hidden_size':3584,
                    'features':feats,'labels':labels,'depths':labels+1,
                    'sample_ids':[r.get('sample_id','') for r in batch],
                    'root_ids':[r.get('root_id','') for r in batch],
                    'family_ids':[r.get('family_id','') for r in batch],
                    'state_sha256s':[r.get('state_sha256','') for r in batch]},
                   out_path)
        done+=1
        if done % 3 == 0: print('chunks', done, flush=True)
        del enc,out,feats
# write manifest
import math
shards=[]
for i in range(0, n, SNAP):
    p=f'{OUT}/features-{i:06d}-{min(i+SNAP,n):06d}.pt'
    shards.append({'path':os.path.basename(p),'rows':min(SNAP,n-i),'sha256':hashlib.sha256(open(p,'rb').read()).hexdigest()})
json.dump({'schema_version':'new_value_head.feature-manifest.v2','rows':n,'rows_per_shard':SNAP,
           'model_fingerprint':{'sha256':'recorded-at-feature-time'},'shards':shards}, open(f'{OUT}/manifest.json','w'))
print('DONE manifest rows', n)
