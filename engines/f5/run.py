"""F5-TTS engine: run.py --text-file T --ref R.wav --ref-text "..." --out O.wav [--pause s] [--seed n] [--extra json]"""
import os, sys, argparse, json, contextlib, random

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-file", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--ref-text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pause", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra", default="{}")
    a = p.parse_args()
    extra = json.loads(a.extra or "{}")
    if not a.ref_text.strip():
        sys.exit("--ref-text must be non-empty (ASR fallback disabled offline)")

    # library prints progress to stdout; route all of it to stderr
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np, soundfile as sf, torch
        from importlib.resources import files
        from omegaconf import OmegaConf
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import (load_vocoder, load_model,
                                              preprocess_ref_audio_text, infer_process)

        device = "cpu"  # MPS hung at first sampling step on M4 Max; opt in via --extra device=mps
        device = extra.pop("device", device)
        seed = a.seed or random.randint(1, 2**31 - 1)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
        vocoder = load_vocoder("vocos", is_local=True, local_path=os.path.join(W, "vocos"), device=device)
        model = load_model(DiT, cfg.model.arch,
                           os.path.join(W, "F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"),
                           mel_spec_type="vocos",
                           vocab_file=os.path.join(W, "F5-TTS/F5TTS_v1_Base/vocab.txt"),
                           device=device)
        ref_audio, ref_text = preprocess_ref_audio_text(a.ref, a.ref_text)

        text = open(a.text_file, encoding="utf-8").read()
        paras = [t.strip() for t in text.split("\n\n") if t.strip()]
        if not paras:
            sys.exit("empty text")
        kw = {k: extra[k] for k in ("nfe_step", "cfg_strength", "sway_sampling_coef",
                                     "speed", "cross_fade_duration", "target_rms") if k in extra}
        chunks, sr = [], 24000
        for t in paras:
            wav, sr, _ = infer_process(ref_audio, ref_text, t, model, vocoder,
                                       mel_spec_type="vocos", device=device, show_info=lambda *x: None, **kw)
            chunks.append(np.asarray(wav, dtype=np.float32))
            chunks.append(np.zeros(int(a.pause * sr), dtype=np.float32))
        sf.write(a.out, np.concatenate(chunks[:-1]), sr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr)
        sys.exit(1)
