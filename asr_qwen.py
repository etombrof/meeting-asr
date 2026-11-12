"""French words + timestamps with Qwen3-ASR 1.7B -> <run>/words.json"""
import json, sys, time, unicodedata
import torch
from qwen_asr import Qwen3ASRModel


def punctuated(text, stamps):
    """Qwen's own text with the aligner's times. The aligner keeps only letters, digits and ' of each
    space-separated piece ("2,5" -> "25", "l’a" -> "la", "oui." -> "oui"), so each piece of the text is
    matched to the 1+ tokens made from it (2 when chunk texts got glued together)."""
    keep = lambda s: "".join(c for c in s if c == "'" or unicodedata.category(c)[0] in "LN")
    out, i = [], 0
    for seg in text.split():
        k = keep(seg)
        if not k:  # punctuation alone, e.g. the "?" in "quoi ?"
            if out:
                out[-1]["word"] += " " + seg
            continue
        j, got = i, ""
        while len(got) < len(k):
            got += stamps[j].text
            j += 1
        if got != k:
            raise ValueError(f"text and timestamps disagree at {seg!r}: {got!r}")
        out.append({"start": stamps[i].start_time, "end": stamps[j - 1].end_time, "word": seg})
        i = j
    assert i == len(stamps), f"{len(stamps) - i} timestamps left over"
    return out


if __name__ == "__main__":
    wav, lang, run = sys.argv[1], sys.argv[2], sys.argv[3]
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B", forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        # ASR (4.4 GB) + aligner (1.8 GB) overflow the 6 GB GPU, so the aligner runs on CPU
        forced_aligner_kwargs={"dtype": torch.float32, "device_map": "cpu"},
        max_inference_batch_size=1,
        # per 3-min chunk; the default 512 cut ~1 min off the end of every chunk of normal-pace French
        max_new_tokens=2048,
        dtype=torch.bfloat16, device_map="cuda")
    t = time.time()
    r = model.transcribe(wav, language=lang, return_time_stamps=True)[0]
    try:
        words = punctuated(r.text, r.time_stamps.items)
    except (ValueError, IndexError, AssertionError) as e:  # keep the meeting, lose punctuation
        print(f"warning: {e}; using unpunctuated words", flush=True)
        words = [{"start": w.start_time, "end": w.end_time, "word": w.text} for w in r.time_stamps]
    json.dump({"seconds": time.time() - t, "words": words},
              open(f"{run}/words.json", "w"), indent=1, ensure_ascii=False)
    print(f"{len(words)} words")
