# F5-TTS engine

Install (f5-tts 1.1.22): `uv venv -p 3.11 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install f5-tts`

## Weights (fetched with git clone + git lfs, loaded from local paths, HF_HUB_OFFLINE=1)
- weights/F5-TTS: SWivid/F5-TTS @ 84e5a410d9cead4de2f847e7c9369a6440bdfaca (only F5TTS_v1_Base/model_1250000.safetensors + vocab.txt)
- weights/vocos: charactr/vocos-mel-24khz @ 0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21

## License
- Code: MIT (f5-tts)
- F5-TTS weights: CC-BY-NC-4.0 (non-commercial, because of the Emilia training data)
- Vocos weights: MIT

## Device
CPU is the default. On the M4 Max, MPS hung at the first sampling step (more than 20 min stuck at 0%). To try it anyway: --extra '{"device":"mps"}'.

## Usage
.venv/bin/python run.py --text-file T.txt --ref REF.wav --ref-text "transcript" --out OUT.wav [--pause 0.4] [--seed 0] [--extra JSON]
- --ref-text is required. Whisper ASR fallback is disabled because it would need another download.
- The ref is clipped to 12 s internally and resampled to 24 kHz. Output is 24 kHz mono.

## --extra keys
device (cpu|mps), nfe_step (default 32; 16 = about 2x faster), cfg_strength (2.0), sway_sampling_coef (-1.0), speed (1.0), cross_fade_duration (0.15), target_rms (0.1)
