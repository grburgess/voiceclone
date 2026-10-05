# Qwen3-TTS engine (0.6B Base, voice clone)

- Code: `qwen-tts` 0.1.1 (PyPI, github.com/QwenLM/Qwen3-TTS), Apache-2.0.
- Weights: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (includes `speech_tokenizer/`), Apache-2.0.
  Cloned from ModelScope mirror (huggingface.co unreachable via sandbox proxy):
  `https://www.modelscope.cn/Qwen/Qwen3-TTS-12Hz-0.6B-Base.git`, commit `fda1995a3162ad0488393cc366b82ca59c91c08e`.
- Device: MPS (Apple M4 Max), float32. Output 24 kHz mono.
- Ref audio resampled to 24 kHz internally (librosa).

## Modes
- `--ref-text` given -> ICL clone (ref audio + transcript). Transcript MUST match ref audio; a wrong transcript makes generation run away to max_new_tokens.
- `--ref-text` empty -> speaker-embedding (x-vector) only clone.

## --extra keys
`language` (default "Auto"; e.g. "English"), `device` ("mps"/"cpu"), `dtype` ("bf16"),
and sampling knobs passed to generate: `do_sample`, `temperature` (0.9), `top_k` (50), `top_p` (1.0),
`repetition_penalty` (1.05), `subtalker_dosample`, `subtalker_temperature`, `subtalker_top_k`, `subtalker_top_p`, `max_new_tokens` (2048).

Note: harmless "SoX could not be found" / flash-attn warnings go to stderr.
