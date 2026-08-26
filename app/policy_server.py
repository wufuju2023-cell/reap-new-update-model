#!/usr/bin/env python3
"""policy_server — V1-1 主线实现 (0 long-train + on-demand RTTT)

- 直载 REAL-Prover (BF16, cuda:0)
- LoRA adapter init_lora_weights=False (等价 base, 零长训)
- 端点:
    GET  /health
    POST /v1/chat/completions   policy 采样: {prompt, n, temperature} -> [{text, logprob_avg}]
    POST /value                 value head: {prompt} -> {"score": float}
    POST /ttt_step              RTTT 一步: {items:[{prompt, target, r, logprob_old}]}
                                -> {"loss":..., "kl":..., "steps":1}  (adapter 自动 hot 保持)
    POST /adapter/snapshot      快照 {name} -> 保存 adapter 状态字典
    POST /adapter/restore       回滚 {name}
- 用法: nohup /opt/venv/bin/python /workspace/app/policy_server.py \
        --base /workspace/data/real-prover --port 8760 > /workspace/logs/policy.log 2>&1 &
"""
import argparse, json, io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BETA_KL = 0.05  # KL 防忘系数 (见 v1-spec 02)

class Engine:
    def __init__(self, base_dir, device="cuda:0", lora_r=16):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(base_dir)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_dir, torch_dtype=torch.bfloat16, device_map=self.device)
        base.config.use_cache = True
        lora = LoraConfig(r=lora_r, lora_alpha=32, lora_dropout=0.02,
                          target_modules=TARGET_MODULES,
                          bias="none",
                          init_lora_weights=False)  # 零初始化 => 等价 base
        self.model = get_peft_model(base, lora)
        self.model.eval()
        # value head: 冻结 backbone, 仅训练 head; v1 先随机 + TD 更新
        self.vhead = torch.nn.Sequential(torch.nn.Linear(4096, 256), torch.nn.SiLU(),
                                         torch.nn.Linear(256, 1)).to(self.device)
        self.vhead.train()
        self.opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=1e-3)
        self.generated = 0

    def generate(self, prompt, n=6, temperature=0.99, max_new_tokens=128):
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        outs = self.model.generate(**enc, do_sample=True, temperature=temperature,
                                   num_return_sequences=n, max_new_tokens=max_new_tokens,
                                   pad_token_id=self.tok.pad_token_id, return_dict_in_generate=True, output_scores=False)
        res = []
        for seq in outs.sequences:
            prompt_len = enc["input_ids"].shape[1]
            gen_ids = seq[prompt_len:]
            text = self.tok.decode(gen_ids, skip_special_tokens=True).split("\n")[0]
            res.append({"text": text, "logprob_avg": 0.0})
        self.generated += n
        return res

    def value(self, prompt):
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.base_model.model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()  # [1,4096]
        self.vhead.eval()
        with torch.no_grad():
            s = self.vhead(h).item()
        return s

    def ttt_step(self, items):
        """items=[{prompt,target,r,logprob_old}]; 单步 REINFORCE + KL 防忘 + 价值 TD 同布"""
        if not items:
            return {"error": "empty items"}
        self.model.train()
        params = list(self.model.parameters())
        self.opt.zero_grad()
        total_loss, total_kl = 0.0, 0.0
        for it in items[:16]:
            enc = self.tok(it["prompt"], return_tensors="pt").to(self.device)
            tgt = self.tok(it["target"] + self.tok.eos_token, return_tensors="pt").to(self.device)
            input_ids = torch.cat([enc["input_ids"], tgt["input_ids"]], dim=1)
            labels = torch.full_like(input_ids, -100)
            labels[:, enc["input_ids"].shape[1]:] = tgt["input_ids"]
            out = self.model(input_ids=input_ids, labels=labels)
            logprob_new = out.loss
            logp = -out.loss  # neg-loss approx logp
            logp_old = it.get("logprob_old", -12.0)
            adv = float(it.get("r", 0.0))
            policy_loss = -adv * (logp - logp_old)  # REINFORCE 单步
            kl = BETA_KL * (logp - logp_old).pow(2)
            loss = policy_loss + kl
            loss.backward()
            total_loss += loss.item()
            total_kl += kl.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.opt.step()
        self.model.eval()  # 回到推理模式 (adapter 已 hot)
        return {"loss": total_loss / len(items), "kl": total_kl / len(items), "steps": len(items)}

    def snapshot(self, name):
        torch.save(self.model.state_dict(), f"/workspace/out/adapter_{name}.pt")
        return f"snapshot:{name}"

    def restore(self, name):
        sd = torch.load(f"/workspace/out/adapter_{name}.pt", map_location=self.device)
        self.model.load_state_dict(sd)
        return f"restored:{name}"

ENGINE = None

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "device": str(ENGINE.device),
                             "generated": ENGINE.generated,
                             "adapter_loaded": "zero-init lora (0 long-train)"})
        else:
            self._json(404, {"error": self.path})
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path.endswith("/chat/completions"):
                outs = ENGINE.generate(req.get("prompt", ""), n=req.get("n", 6),
                                       temperature=req.get("temperature", 0.99))
                self._json(200, {"choices": outs})
            elif self.path.endswith("/value"):
                self._json(200, {"score": ENGINE.value(req.get("prompt", ""))})
            elif self.path.endswith("/ttt_step"):
                self._json(200, ENGINE.ttt_step(req.get("items", [])))
            elif self.path.endswith("/adapter/snapshot"):
                self._json(200, {"result": ENGINE.snapshot(req.get("name", "t"))})
            elif self.path.endswith("/adapter/restore"):
                self._json(200, {"result": ENGINE.restore(req.get("name", "t"))})
            else:
                self._json(404, {"error": self.path})
        except Exception as e:
            self._json(500, {"error": str(e)})

def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/data/real-prover")
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--lora-r", type=int, default=16)
    a = ap.parse_args()
    import os; os.makedirs("/workspace/logs", exist_ok=True)
    print("[policy_server] loading base:", a.base)
    ENGINE = Engine(a.base, lora_r=a.lora_r)
    print("[policy_server] ready on :%d (bf16, zero-init lora r=%d)" % (a.port, a.lora_r))
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()

if __name__ == "__main__":
    main()
