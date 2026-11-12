"""Give each word the speaker who talks most during it -> <run>/transcript.md"""
import json, sys

run = sys.argv[1]
words = json.load(open(f"{run}/words.json"))["words"]
turns = json.load(open(f"{run}/diarization.json"))["turns"]
labels = {s: f"Speaker {i + 1}" for i, s in enumerate(dict.fromkeys(t["speaker"] for t in turns))}


def speaker(w):
    if not turns:
        return "Speaker 1"
    overlap = lambda t: min(w["end"], t["end"]) - max(w["start"], t["start"])
    best = max(turns, key=overlap)
    if overlap(best) <= 0:  # word falls in a gap: take the closest turn
        mid = (w["start"] + w["end"]) / 2
        best = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))
    return labels[best["speaker"]]


lines = []  # [start, speaker, [words]]
for w in words:
    s = speaker(w)
    if lines and lines[-1][1] == s:
        lines[-1][2].append(w["word"])
    else:
        lines.append([w["start"], s, [w["word"]]])

with open(f"{run}/transcript.md", "w") as f:
    for start, s, ws in lines:
        f.write(f"**[{int(start // 60):02d}:{int(start % 60):02d}] {s}:** {' '.join(ws)}\n\n")
print(f"{len(lines)} speaker turns")
