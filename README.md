# voiceclone

Local voice-cloning text-to-speech for macOS (Apple Silicon). Six open models behind one web UI and one command,
plus reference-clip cleanup and Voice Memos import. Everything runs offline once installed.

## Install

```sh
brew install uv ffmpeg git-lfs && git lfs install
./install.sh            # everything (~65 GB); or pick: ./install.sh main whisper qwen3
```

Each model gets its own venv (pinned in `requirements/`) and weights pinned to a commit. Re-running is safe;
it skips what is already there. Needs a network that can reach huggingface.co, modelscope.cn and GitHub.
`voiceclone` is linked into `~/.local/bin`.

## Use

```sh
voiceclone ui                       # web UI, opens browser, auto-stops after 3h (--hours N)
voiceclone stop
voiceclone say "Hello there."       # qwen3 + voices/me.wav -> outputs/
voiceclone say -m fish -v me -f script.txt -o narration.wav -s 7
voiceclone voices                   # * = has transcript
voiceclone transcribe me            # redo voices/me.txt with Whisper
```

Script files: a blank line starts a new paragraph. Each paragraph is generated separately and joined with a pause.

## Models

| key | license | speed (one sentence, incl. load) | transcript | notes |
|---|---|---|---|---|
| `chatterbox` | MIT | fast | not used | weak accent; exaggeration/cfg sliders |
| `qwen3` | Apache-2.0 | ~15 s | optional, better with | default |
| `voxcpm` | Apache-2.0 | ~25 s | optional | |
| `cosyvoice` | Apache-2.0 | ~30 s (CPU) | required | |
| `fish` | **non-commercial** | ~70 s | required | likely best accent; ref 10–30 s |
| `f5` | **non-commercial** | ~60 s (CPU) | required | uses first 12 s of ref; aim 6–10 s |

Every engine follows one contract (`engines/<key>/run.py`):
`run.py --text-file T --ref REF.wav --ref-text "..." --out OUT.wav [--pause S] [--seed N] [--extra JSON]`.
Per-engine `--extra` keys and weight commits are in `engines/<key>/README.md`.

## Voices

`voices/<name>.wav` is the reference clip; `voices/<name>.txt` its transcript (Whisper-generated, edit to fix
mistakes — it must match the clip exactly). Add voices in the UI ("Add a new voice"): pick a Voice Memo,
trim, optionally clean up (DeepFilterNet, Resemble Enhance, VoiceFixer, ClearerVoice, Demucs isolation).

## Layout

```
app.py        web UI (Gradio)          tts.py      Chatterbox CLI
cleanup.py    reference-clip cleanup   bin/        voiceclone CLI
engines/      one folder per model     requirements/  pinned venvs
voices/ outputs/                       (user data, not tracked)
```
