# CosyVoice 3 engine

- Code: FunAudioLLM/CosyVoice @ `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` (Apache-2.0), in `src/`
- Weights: HF `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` @ `29e01c4e8d000f4bcd70751be16fa94bf3d85a18` (Apache-2.0), in `weights/Fun-CosyVoice3-0.5B`. LFS files pulled: llm.pt, flow.pt, hift.pt, campplus.onnx, speech_tokenizer_v3.onnx, CosyVoice-BlankEN/model.safetensors. Skipped: llm.rl.pt, flow.decoder.estimator.fp32.onnx (TensorRT only), speech_tokenizer_v3.batch.onnx.
- Text normaliser: wetext FSTs, from a git clone of modelscope `pengzhendong/wetext` @ `030bb1febce0bd0168549a03e181cd9c1d70c799`, in `weights/wetext`. run.py patches `snapshot_download` so the FSTs load from that local folder. ttsfrd is not installed.
- Device: CPU. Upstream code picks cuda or cpu only, so MPS isn't used. No TensorRT, vllm or deepspeed.
- Output: 24 kHz mono WAV. Load time is about 40 s. Generation runs at about 3.6x real time on an M4 Max.
- The `--ref-text` transcript is required for zero-shot cloning. run.py adds the prefix `You are a helpful assistant.<|endofprompt|>` to it.

## Usage
```
.venv/bin/python run.py --text-file T.txt --ref REF.wav --ref-text "transcript" --out OUT.wav [--pause 0.4] [--seed 0] [--extra '{...}']
```
Paragraphs are separated by blank lines. `--seed 0` picks a random seed. For paragraph i, the seed is seed+i.

## --extra keys
| key | default | meaning |
|---|---|---|
| speed | 1.0 | speaking rate |
| text_frontend | true | wetext normalisation and splitting |
| system_prompt | "You are a helpful assistant." | text placed before `<|endofprompt|>` in the prompt |
| threads | cpu_count | torch CPU threads |
| verbose | false | send library logs and progress to stderr |
