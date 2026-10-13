# Auto-tag engine
- Adds Fish S2 performance tags (`[sigh]`, `[whisper]`, free-form directions) to a plain script. Not a TTS engine; output is text.
- Code: pip `mlx-lm` 0.32.0, MIT. Weights: huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit, Apache-2.0, commit 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b (ModelScope mirror mlx-community/Qwen3-4B-Instruct-2507-4bit @ ec7ca96 has byte-identical model.safetensors/tokenizer), in `weights/qwen3-4b` (2.3G model, 4.3G on disk with the git-lfs store).
- Device: Apple Silicon GPU via MLX. Greedy decoding, one paragraph per call.
- Dialogue `name:` prefixes are stripped before the LLM and re-attached after (a 4B model mangles them).
- Guard: same line count, and with tags stripped the output words must equal the input words, else exit 1.
- Usage: `.venv/bin/python run.py --text-file T.txt` -> tagged script on stdout.
