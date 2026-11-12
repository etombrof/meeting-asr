"""English words + timestamps with Parakeet TDT 0.6B v2 -> <run>/words.json
(v3 silently translates French to English, so French goes to asr_qwen.py)"""
import json, sys, time
import numpy as np, soundfile as sf
import torch
import nemo.collections.asr as nemo_asr

SR, PIECE = 16000, 180 * 16000  # memory grows with length (31 min OOMs on 6 GB; 5 min nearly), so ~3-min pieces


def cuts(audio):
    """Piece boundaries, each moved to the quietest 0.1 s within 2 s of the target so no word is split."""
    c = [0]
    while len(audio) - c[-1] > PIECE + 10 * SR:  # no tiny last piece
        lo = c[-1] + PIECE - 2 * SR
        frames = audio[lo: lo + 4 * SR].reshape(-1, 1600)
        c.append(lo + int(np.argmin((frames ** 2).mean(1))) * 1600 + 800)
    c.append(len(audio))
    assert all(0 < b - a <= PIECE + 12 * SR for a, b in zip(c, c[1:])), c
    return c


def transcribe(a, b):
    """Words of audio[a:b]. Parakeet sometimes drops all punctuation/capitals for a whole piece,
    depending only on how it starts (2% vs 17% punctuated for a 1 s shift), so such pieces
    are re-run with 1-3 s of silence in front; the best-punctuated try wins."""
    tries = []
    for pad in (0, 1, 2, 3):
        x = np.concatenate([np.zeros(pad * SR, dtype=np.float32), audio[a:b]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = model.transcribe([x], timestamps=True, batch_size=1, verbose=False)[0]
        off = a / SR - pad
        ws = [{"start": w["start"] + off, "end": w["end"] + off, "word": w["word"]} for w in r.timestamp["word"]]
        punct = sum(w["word"][-1] in ".,?!" for w in ws) / max(len(ws), 1)
        tries.append((punct, ws))
        if punct >= 0.10 or len(ws) < 50:  # normal pieces: 10-19% of words end with punctuation
            break
        print(f"  {punct:.0%} punctuated" + (f", retrying with {pad + 1} s of silence in front" if pad < 3 else ""),
              flush=True)
    return max(tries, key=lambda t: t[0])[1]


wav, run = sys.argv[1], sys.argv[2]
audio = sf.read(wav, dtype="float32")[0]
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").cuda().eval()
# full attention on 3-min pieces: local attention dropped most punctuation in some pieces
model.change_subsampling_conv_chunking_factor(8)  # "auto" (1) OOMs on 6 GB
t, words, c = time.time(), [], cuts(audio)
for i, (a, b) in enumerate(zip(c, c[1:])):
    print(f"progress: piece {i + 1}/{len(c) - 1} · {a / SR / 60:.0f}–{b / SR / 60:.0f} min", flush=True)
    words += transcribe(a, b)
json.dump({"seconds": time.time() - t, "words": words},
          open(f"{run}/words.json", "w"), indent=1, ensure_ascii=False)
print(f"{len(words)} words")
