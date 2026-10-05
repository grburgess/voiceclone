import os, sys, argparse, json
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights", "VoxCPM2")

def main():
    sys.stdout = sys.stderr
    p = argparse.ArgumentParser()
    p.add_argument("--text-file", required=True); p.add_argument("--ref", required=True)
    p.add_argument("--ref-text", default=None); p.add_argument("--out", required=True)
    p.add_argument("--pause", type=float, default=0.4); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra", default="{}")
    a = p.parse_args()
    extra = json.loads(a.extra)
    import torch
    from voxcpm import VoxCPM
    if a.seed: torch.manual_seed(a.seed); np.random.seed(a.seed)
    m = VoxCPM.from_pretrained(WEIGHTS, load_denoiser=False, optimize=False,
                               device=extra.pop("device", None))
    sr = m.tts_model.sample_rate
    paras = [t.strip() for t in open(a.text_file).read().split("\n\n") if t.strip()]
    gap = np.zeros(int(a.pause * sr), dtype=np.float32); out = []
    for i, t in enumerate(paras):
        kw = dict(text=t, reference_wav_path=a.ref, **extra)
        if a.ref_text:  # ultimate cloning: ref also as continuation prompt
            kw.update(prompt_wav_path=a.ref, prompt_text=a.ref_text)
        out += ([gap] if i else []) + [np.asarray(m.generate(**kw), dtype=np.float32)]
    sf.write(a.out, np.concatenate(out), sr)

if __name__ == "__main__":
    try: main()
    except Exception as e: print(f"voxcpm error: {e!r}", file=sys.stderr); sys.exit(1)
