"""Print 'fr' or 'en' for a wav, from 30 s in the middle (Whisper large-v3).
Meetings are assumed to be English or French only; anything else -> en."""
import sys
from faster_whisper import WhisperModel, decode_audio

audio = decode_audio(sys.argv[1])
mid = len(audio) // 2
sample = audio[max(0, mid - 15 * 16000): mid + 15 * 16000]
lang, prob, _ = WhisperModel("large-v3", device="cuda", compute_type="float16").detect_language(sample)
print("fr" if lang == "fr" and prob > 0.5 else "en")
