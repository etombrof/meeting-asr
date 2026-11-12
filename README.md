# meeting-asr

Press a hotkey to record a meeting on Linux (Teams, Meet, any app, or people in the room). It produces a Markdown note with a short summary and a transcript labelled by speaker, for example in Obsidian.
Speech-to-text and speaker separation run locally on an NVIDIA GPU. Only the summary step uses a cloud service: Claude, through the `claude` CLI.

```
hotkey ─► recorder.py ─► recordings/<date>.wav    left = your mic, right = what you hear
                              │
          process ◄───────────┘                   (any audio/video file also works)
            language: Whisper ─► words: Parakeet v2 (EN) / Qwen3-ASR (FR) ─► speakers: pyannote
            ─► merge ─► summary: claude -p ─► $NOTES_DIR/<date>.md
```

| Meeting | language | words | speakers | summary | total |
|---|---|---|---|---|---|
| 31 min, English | 3 s | 28 s | 100–140 s | 11 s | ~3 min |
| 20 min, French | 3–8 s | 303 s | 67 s | 11 s | ~6.5 min |

Tested on Ubuntu 22.04, GNOME on X11, PipeWire 0.3.48 and an RTX 3060 Laptop GPU with 6 GB.

## Requirements

| What | For |
|---|---|
| Linux with PipeWire or PulseAudio (`parec`, `pactl`) | recording |
| NVIDIA GPU ≥ 6 GB, `nvidia-smi` | models, GPU panel |
| GNOME Terminal | recorder window |
| `ffmpeg`, `notify-send` | audio conversion, "note ready" pop-up |
| system `python3` 3.10–3.12 with `faster-whisper` | recorder (uses `audioop`, removed in 3.13), language check |
| [`uv`](https://docs.astral.sh/uv/) | the 2 model environments |
| [`claude` CLI](https://github.com/anthropics/claude-code), logged in | summary |
| Hugging Face account, terms accepted for [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | speaker model (gated) |

Disk: about 7 GB for the environments, plus about 12 GB of models downloaded on first use.

## Setup

```sh
git clone https://github.com/etombrof/meeting-asr && cd meeting-asr
sudo apt install ffmpeg pulseaudio-utils libnotify-bin
pip install --user faster-whisper==1.2.1
uv venv -p 3.12 .venv-qwen && VIRTUAL_ENV=.venv-qwen uv pip install qwen-asr==0.0.6 pyannote.audio==4.0.7
uv venv -p 3.12 .venv-nemo && VIRTUAL_ENV=.venv-nemo uv pip install "nemo_toolkit[asr]==3.0.0"
.venv-qwen/bin/hf auth login                                  # after accepting the pyannote terms
echo 'NOTES_DIR=$HOME/Obsidian/<vault>/Meetings' > config      # default: ~/Meetings
ln -s "$PWD/meeting-rec" ~/.local/bin/
```

Hotkey: GNOME Settings → Keyboard → Custom Shortcuts. Set the command to `/home/<you>/.local/bin/meeting-rec` and pick a shortcut, for example Super+Shift+M.
In Teams, Meet and other apps, set the mic and speaker to "Default": the recorder records the system defaults and follows them when a headset is plugged in.

## Use

| To | Do |
|---|---|
| Start / stop recording | hotkey (press again to stop), or Enter in the window |
| Keep your voice out of the recording | `m` (the meeting app still hears you) |
| Watch processing | the window stays open: steps, GPU, `l` for the live log |
| Open the note in Obsidian | `o` |
| Close the window | `q` (processing continues) |
| Process an existing file | `./process <audio or video file>` |

## Notes

- **Muting in Teams or Meet doesn't stop the system mic.** Your voice is still recorded; use `m` instead.
- **Parakeet TDT v3 lists French, but in our test it output an English translation of French audio.** So v2 handles English and Qwen3-ASR handles French. The language is detected once per meeting, so a meeting that mixes languages is transcribed as one of them.
- **Parakeet sometimes drops all punctuation in a 3-min piece,** depending on where the piece starts (2% vs 17% of words punctuated for a 1 s shift). Those pieces are re-run with 1–3 s of silence in front.
- **On PipeWire 0.3.48, `@DEFAULT_MONITOR@` records the mic.** The recorder names the output's `.monitor` source directly instead.
- **Privacy:** the transcript is sent to Anthropic for the summary; nothing else leaves the machine. Tell people they are being recorded, since consent rules differ by country.

## Files

| File | Does |
|---|---|
| `meeting-rec` | hotkey target: opens the recorder, or stops the one that is recording |
| `recorder.py` | window: records, starts `process`, shows progress |
| `process` | runs the steps; progress in `runs/<name>/progress.log`, step logs next to it |
| `lang.py` | `en` or `fr`, from 30 s in the middle |
| `asr_parakeet.py`, `asr_qwen.py` | words with timestamps |
| `diarize.py` | who spoke when |
| `merge.py` | words + speakers → `transcript.md` |
| `config` | your `NOTES_DIR` (not committed) |
