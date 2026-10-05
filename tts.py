"""Text -> WAV with Chatterbox. Generates per paragraph, stitches with short pauses.

Usage:
  python tts.py script.txt -o out.wav [--voice ref.wav] [--exaggeration 0.5] [--cfg 0.5]
"""
import argparse
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

p = argparse.ArgumentParser()
p.add_argument("text_file")
p.add_argument("-o", "--out", default="out.wav")
p.add_argument("--voice", help="5-10s reference clip to clone")
p.add_argument("--exaggeration", type=float, default=0.5)
p.add_argument("--cfg", type=float, default=0.5)
p.add_argument("--pause", type=float, default=0.4, help="seconds between paragraphs")
a = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
# checkpoints saved on CUDA; remap on load
_load = torch.load
torch.load = lambda *x, **k: _load(*x, **{**k, "map_location": device})
model = ChatterboxTTS.from_pretrained(device=device)

paras = [s.strip() for s in Path(a.text_file).read_text().split("\n\n") if s.strip()]
gap = torch.zeros(1, int(model.sr * a.pause))
chunks = []
for i, para in enumerate(paras, 1):
    print(f"[{i}/{len(paras)}] {para[:60]}")
    chunks += [model.generate(para, audio_prompt_path=a.voice,
                              exaggeration=a.exaggeration, cfg_weight=a.cfg).cpu(), gap]
ta.save(a.out, torch.cat(chunks[:-1], dim=1), model.sr)
print("wrote", a.out)
