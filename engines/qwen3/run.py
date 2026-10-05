import os, sys, json, argparse, random, warnings, io, contextlib
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights", "base")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--pause", type=float, default=0.4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default="{}")
    a = ap.parse_args()
    extra = json.loads(a.extra)
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np, torch, soundfile as sf, librosa
        from qwen_tts import Qwen3TTSModel
        seed = a.seed or random.randint(1, 2**31 - 1)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        dev = extra.pop("device", "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = {"mps": torch.float32, "cpu": torch.float32}.get(dev, torch.bfloat16)
        if extra.pop("dtype", None) == "bf16": dtype = torch.bfloat16
        lang = extra.pop("language", "Auto")
        model = Qwen3TTSModel.from_pretrained(WEIGHTS, device_map=dev, dtype=dtype, attn_implementation="sdpa")
        ref, rsr = librosa.load(a.ref, sr=24000, mono=True)
        xvec = not a.ref_text.strip()
        prompt = model.create_voice_clone_prompt(ref_audio=(ref, rsr), ref_text=a.ref_text.strip() or None, x_vector_only_mode=xvec)
        paras = [p.strip() for p in open(a.text_file).read().split("\n\n") if p.strip()]
        if not paras: raise SystemExit("empty text")
        out, sr = [], None
        for i, p in enumerate(paras):
            wavs, sr = model.generate_voice_clone(text=p, language=lang, voice_clone_prompt=prompt, **extra)
            if i: out.append(np.zeros(int(a.pause * sr), dtype=np.float32))
            out.append(np.asarray(wavs[0], dtype=np.float32))
        sf.write(a.out, np.concatenate(out), sr)

if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr); sys.exit(1)
