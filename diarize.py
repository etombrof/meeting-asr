"""Who spoke when: pyannote community-1 -> <run>/diarization.json
Usage: diarize.py <wav> <run> [n_speakers]   (count is estimated when omitted)"""
import json, sys, time
import torch
from pyannote.audio import Pipeline

wav, run = sys.argv[1], sys.argv[2]
n = {"num_speakers": int(sys.argv[3])} if len(sys.argv) > 3 else {}
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1").to(torch.device("cuda"))
shown = {}


def hook(step, artifact, file=None, total=None, completed=None):
    """pyannote calls this as it works; print 'progress: <stage> <pct>%' every 5% (read by recorder.py)."""
    pct = min(100, int(100 * completed / total)) if total else None  # segmentation overshoots
    last = shown.get(step)
    if step in shown and (pct is None or (last is not None and (pct == last or (pct - last < 5 and pct != 100)))):
        return
    shown[step] = pct
    print(f"progress: {step}" + (f" {pct}%" if pct is not None else ""), flush=True)


t = time.time()
out = pipe(wav, hook=hook, **n)
turns = [{"start": s.start, "end": s.end, "speaker": spk}
         # exclusive = one speaker at a time, meant for aligning with transcripts
         for s, _, spk in out.exclusive_speaker_diarization.itertracks(yield_label=True)]
json.dump({"seconds": time.time() - t, "turns": turns}, open(f"{run}/diarization.json", "w"), indent=1)
print(f"{len({t['speaker'] for t in turns})} speakers")
