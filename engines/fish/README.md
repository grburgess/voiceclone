# Fish Audio S2 Pro engine

- Code: github.com/fishaudio/fish-speech @ 214da3c (v2.0.0), in `src/`. License: Fish Audio Research License.
- Weights: huggingface.co/fishaudio/s2-pro @ 1de9996b6be38b745688de084d87a5633f714e4e (public, no auth), in `weights/s2-pro/`. License: Fish Audio Research License — research/non-commercial free; commercial use needs a license from Fish Audio.
- Device: MPS (Apple M4 Max), bf16; CPU fallback if MPS unavailable. Output 44.1 kHz mono WAV.
- Offline: HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1; all files load from local paths (LLM, tokenizer, codec.pth).
- Install: `uv venv -p 3.11 .venv`; deps from src/pyproject.toml minus pyaudio (needs portaudio), gradio, wandb, tensorboard; then `uv pip install --no-deps -e src`.
- --ref-text required (S2 Pro conditions on ref audio+transcript); wrong transcript degrades cloning.
- Ref encoded once (engine memory cache); any sample rate/format accepted (resampled internally).

Usage:
`.venv/bin/python run.py --text-file T.txt --ref REF.wav --ref-text "..." --out OUT.wav [--pause 0.4] [--seed 0] [--extra '{"temperature":0.8}']`

Multi-speaker: repeat `--ref R --ref-text T` per speaker; reference i is `<|speaker:i|>` in the text (`dialogue.py` writes these from `name: line` scripts). All speakers are generated in one pass per paragraph.

Inline tags (passed through untouched; the model reads them): `[pause]` `[short pause]` `[emphasis]` `[sigh]` `[inhale]` `[exhale]` `[laughing]` `[chuckle]` `[whisper]` `[low voice]` `[excited]` `[surprised]` `[sad]` `[angry]` `[shouting]` `[volume up]` `[volume down]` `[clearing throat]` and free-form `[warm, like talking to an old friend]` (S2 claims 15k+ tags). Drama 3's word-scoped `<whisper>…</whisper>` is API-only, not S2.

--seed 0 = random; paragraph i uses seed+i. Set FISH_TIMING=1 to print load/gen timing to stderr.

--extra keys (ServeTTSRequest fields): temperature (0.8, 0.1–1.0), top_p (0.8, 0.1–1.0), repetition_penalty (1.1, 0.9–2.0), max_new_tokens (1024), chunk_length (200), normalize (true), device ("mps"/"cpu", consumed by loader).

test.wav: made with placeholder ref-text (me.wav not transcribed).
