# VoxCPM engine
- Code: pip `voxcpm` 2.0.3, Apache-2.0. Weights: openbmb/VoxCPM2, Apache-2.0, commit 32279effe8c19989596f05d353d1447f51d9e915, in `weights/VoxCPM2` (model.safetensors 4.3G, audiovae.pth 359M).
- Device: Apple M4 Max MPS, float32 (bf16 auto-promoted). Output 48 kHz mono.
- Offline: HF_HUB_OFFLINE=1; ZipEnhancer/ModelScope denoiser disabled; no other downloads seen.
- --ref-text optional: without it, cloning uses reference audio only; with it, ref is also a continuation prompt (higher fidelity).
- --extra keys (passed to `generate`): cfg_value (2.0), inference_timesteps (10), normalize (false), retry_badcase (true), max_len (4096), device ("mps"/"cpu", consumed by loader).
- Usage: `.venv/bin/python run.py --text-file T.txt --ref REF.wav [--ref-text "..."] --out OUT.wav [--pause 0.4] [--seed 0] [--extra '{"cfg_value":2.0}']`
