"""French words + timestamps with Qwen3-ASR 1.7B -> <run>/words.json"""
import json, sys, time
import torch
from qwen_asr import Qwen3ASRModel

wav, lang, run = sys.argv[1], sys.argv[2], sys.argv[3]
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B", forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    # ASR (4.4 GB) + aligner (1.8 GB) overflow the 6 GB GPU, so the aligner runs on CPU
    forced_aligner_kwargs={"dtype": torch.float32, "device_map": "cpu"},
    max_inference_batch_size=1,
    dtype=torch.bfloat16, device_map="cuda")
t = time.time()
r = model.transcribe(wav, language=lang, return_time_stamps=True)[0]
words = [{"start": w.start_time, "end": w.end_time, "word": w.text} for w in r.time_stamps]
json.dump({"seconds": time.time() - t, "words": words},
          open(f"{run}/words.json", "w"), indent=1, ensure_ascii=False)
print(f"{len(words)} words")
