import json, math, statistics, torch, random, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import peft
BASE='/mnt/workspace/models/REAL-Prover-fe76f68d'
VAL='/mnt/workspace/new_value_head/dataset/validation.jsonl'
SEED=7; rid=sys.argv[1]; head_path=sys.argv[2]
random.seed(SEED); torch.manual_seed(SEED)
rows=[]
with open(VAL) as f:
    for line in f:
        try: rows.append(json.loads(line))
        except: pass
by={}
for r in rows:
    c=r.get('value_class'); by[c]=by.get(c,0)+1
picked=[]
for c,n in sorted(by.items()):
    cand=[r for r in rows if r.get('value_class')==c]
    picked += random.sample(cand, min(12,len(cand)))
labels=[int(r.get('value_class',0))+1 for r in picked]
import io, base64
d=json.load(open('/root/exp1/artifact/backend.json'))
payload=torch.load(io.BytesIO(base64.b64decode(d['payload'])), map_location='cpu', weights_only=True)
dev=torch.device('cuda')
tok=AutoTokenizer.from_pretrained(BASE)
model=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map={'':dev}, use_cache=False); model.eval()
cfg=peft.LoraConfig(r=16,lora_alpha=32,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
pm=peft.get_peft_model(model,cfg); pm.load_state_dict(payload['adapter'],strict=False); pm.eval()
ck=torch.load(head_path, map_location='cpu', weights_only=False)
head=torch.nn.Sequential(torch.nn.Linear(3584,256),torch.nn.SiLU(),torch.nn.Linear(256,64)).to(dev,dtype=torch.float32)
head.load_state_dict(ck['state_dict']); head.eval()
preds=[]
with torch.no_grad():
    for i in range(0,len(picked),12):
        batch=picked[i:i+12]
        enc=tok([r['prompt'] for r in batch], return_tensors='pt', padding=True, truncation=True, max_length=1024).to(dev)
        out=pm(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        lg=head(out.hidden_states[-1][:,-1,:].float())
        support=torch.arange(1,65,device=dev,dtype=torch.float32)
        preds+=(lg.softmax(-1)@support).clamp(1,64).tolist()
def rank(xs):
    idx=sorted(range(len(xs)),key=lambda i:xs[i]); ranks=[0.0]*len(xs); r=0
    while r<len(idx):
        s=r
        while s+1<len(idx) and xs[idx[s+1]]==xs[idx[r]]: s+=1
        avg=(r+s)/2+1
        for k in range(r,s+1): ranks[idx[k]]=avg
        r=s+1
    return ranks
rx,ry=rank(labels),rank(preds); mx,my=statistics.mean(rx),statistics.mean(ry)
num=sum((a-mx)*(b-my) for a,b in zip(rx,ry)); den=math.sqrt(sum((a-mx)**2 for a in rx))*math.sqrt(sum((b-my)**2 for b in ry))
sp=num/den if den else float('nan')
r={'run':rid,'n':len(labels),'spearman':round(sp,4),'mean_pred':round(statistics.mean(preds),3),'mean_true':round(statistics.mean(labels),3),'maxerr':round(max(abs(t-p) for t,p in zip(labels,preds)),3)}
print(json.dumps(r))
json.dump(r, open('/mnt/workspace/head-runs/runs/'+rid+'/e1-eval.json','w'))
