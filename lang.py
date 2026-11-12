"""Print 'fr' or 'en' for a 16 kHz mono wav, from 30 s in the middle (Whisper large-v3).
Meetings are assumed to be English or French only; anything else -> en."""
import sys
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL = "openai/whisper-large-v3"
audio = sf.read(sys.argv[1], dtype="float32")[0]
mid = len(audio) // 2
sample = audio[max(0, mid - 15 * 16000): mid + 15 * 16000]
proc = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
feats = proc(sample, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
langs = model.generation_config.lang_to_id  # {"<|en|>": token id, ...}
start = torch.tensor([[model.generation_config.decoder_start_token_id]], device="cuda")
with torch.no_grad():  # Whisper's first output token is the language; model.detect_language() gives no probability
    logits = model(input_features=feats, decoder_input_ids=start).logits[0, -1]
p = dict(zip(langs, logits[list(langs.values())].float().softmax(0).tolist()))
print(f"French {p['<|fr|>']:.0%}, top {max(p, key=p.get)}", file=sys.stderr)
print("fr" if max(p, key=p.get) == "<|fr|>" and p["<|fr|>"] > 0.5 else "en")
