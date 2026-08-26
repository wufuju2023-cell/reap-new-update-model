import json
import math
import time

import numpy as np
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/workspace/v1/models/REAL-Prover"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

app = Flask(__name__)

print("loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("loading model (bf16)...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    attn_implementation="flash_attention_2",
    # fallback for rocm: sdpa is default; flash_attention_2 works on rocm for 7B qwen
)
model.eval()
print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

CLEANUP = re_compile_safe = None

VALUE_PROMPT = (
    "User: Estimate, on a scale from 0 to 1, the probability that this Lean state can be solved directly by a short tactic, given available theorems. "
    "Respond with ONLY a JSON object of the form {{\"score\": <float>}}. Keep it short.\n\nSTATE:\n{prompt}\n\nVALUE:\n\nAssistant:"
)


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True)
    messages = data.get("messages", [])
    prompt = ""
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        prompt += content

    n = int(data.get("n", 1))
    temperature = float(data.get("temperature", 0.99))
    max_tokens = int(data.get("max_tokens", 1024))
    want_logprobs = bool(data.get("logprobs", False))

    inputs_unbatched = tok(prompt, return_tensors="pt")
    input_len = inputs_unbatched["input_ids"].shape[1]
    input_ids = inputs_unbatched["input_ids"].repeat(n, 1).to(DEVICE)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            do_sample=True,
            temperature=max(temperature, 1e-4),
            top_p=1.0,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.eos_token_id,
        )
        seqs = out.sequences
        scores = out.scores  # list of tensors, length = gen steps

    choices = []
    for i in range(n):
        seq = seqs[i]
        new_ids = seq[input_len:]
        text = tok.decode(new_ids, skip_special_tokens=True)
        logps = []
        if want_logprobs and scores is not None:
            for step, (s, tid) in enumerate(zip(scores, new_ids)):
                logits = s[i]
                logp = torch.log_softmax(logits.float(), dim=-1)[tid].item()
                logps.append(round(float(logp), 6))
        choice = {"index": i, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        if want_logprobs:
            choice["logprobs"] = {
                "content": [{"token": str(tid.item() if torch.is_tensor(tid) else tid), "logprob": lp} for tid, lp in zip(new_ids, logps)]
            }
        choices.append(choice)

    return jsonify({"model": data.get("model", "reap"), "choices": choices}, 200)


@app.route("/value", methods=["POST"])
def value():
    data = request.get_json(force=True)
    state = data.get("state") or ""
    prompt = VALUE_PROMPT.format(prompt=state)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            inputs["input_ids"],
            do_sample=False,
            max_new_tokens=64,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    text = tok.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        score = float(json.loads(text[start:end])["score"])
    except Exception:
        score = 0.5
    score = max(-1.0, min(1.0, score))
    return jsonify({"score": -score})


@app.route("/health")
def health():
    return jsonify({"ok": True, "device": str(DEVICE)})


if __name__ == "__main__":
    # waitress (threaded) if available else flask dev server
    try:
        from waitress import serve

        print("serving with waitress on :8000", flush=True)
        serve(app, host="0.0.0.0", port=8000, threads=8)
    except Exception as e:
        print("waitress unavailable:", e, "-> flask dev", flush=True)
        app.run(host="0.0.0.0", port=8000, threaded=True)
