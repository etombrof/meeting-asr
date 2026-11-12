"""Summary on this machine (SUMMARY=local in config): Qwen3.5-4B, 4-bit on the GPU.
Usage: summarize.py <prompt> < transcript.md > summary.md   (same shape as claude -p)"""
import re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen3.5-4B"  # Apache 2.0, English + French; ~3 GB in 4-bit, so it fits a 6 GB GPU
PIECE = 2048  # tokens read at once: memory grows with this, so a 2-hour meeting (~28k tokens) still fits


def load():
    return AutoTokenizer.from_pretrained(MODEL), AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="cuda", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))


def generate(model, inputs, piece=PIECE, **kw):
    """model.generate, but the prompt is read piece by piece into the cache first. Reading it in one go
    (the default) needs ~0.3 GB per 1k tokens without the optional fast kernels: 28k tokens overflow 6 GB."""
    ids, cache = inputs["input_ids"], None
    with torch.no_grad():
        for a in range(0, ids.shape[1] - 1, piece):  # all but the last token; generate() starts from it
            cache = model(input_ids=ids[:, a:min(a + piece, ids.shape[1] - 1)], past_key_values=cache,
                          use_cache=True, logits_to_keep=1).past_key_values  # all logits: ~1 GB per piece
    return model.generate(**inputs, past_key_values=cache, **kw)


if __name__ == "__main__":
    prompt, transcript = sys.argv[1], sys.stdin.read()
    tok, model = load()
    inputs = tok.apply_chat_template([{"role": "user", "content": f"{prompt}\n\n{transcript}"}],
                                     add_generation_prompt=True, enable_thinking=False,  # thinking is on by default
                                     return_dict=True, return_tensors="pt").to("cuda")
    n, t = inputs["input_ids"].shape[1], time.time()
    # Qwen's recommended non-thinking sampling; greedy decoding makes small models repeat themselves
    out = generate(model, inputs, max_new_tokens=1500, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    text = tok.decode(out[0, n:], skip_special_tokens=True).strip()
    print(re.sub(r"^```(?:markdown)?\n|\n```$", "", text))  # some answers come wrapped in a code fence
    print(f"{n} tokens in, {out.shape[1] - n} out, {time.time() - t:.0f} s, "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", file=sys.stderr)
