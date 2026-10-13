"""Fish Audio S2 Pro engine: text file + ref voice -> one WAV. See README.md."""
import argparse, json, os, sys, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(HERE / "src"))

# keep stdout clean: route all logging to stderr at WARNING+
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-file", required=True)
    p.add_argument("--ref", required=True, action="append")  # repeat --ref/--ref-text per speaker (<|speaker:N|>)
    p.add_argument("--ref-text", required=True, action="append")
    p.add_argument("--out", required=True)
    p.add_argument("--pause", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra", default="{}")
    a = p.parse_args()
    text_file, out, *ref_paths = (Path(x).expanduser().resolve() for x in (a.text_file, a.out, *a.ref))
    if len(a.ref) != len(a.ref_text):
        raise SystemExit("need one --ref-text per --ref")
    extra = json.loads(a.extra)
    os.chdir(HERE / "src")  # repo resolves configs relative to cwd
    real_stdout, sys.stdout = sys.stdout, sys.stderr  # repo prints prompts; keep stdout clean

    import numpy as np, torch, soundfile as sf
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

    import time; t0 = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = extra.pop("device", device)
    w = HERE / "weights" / "s2-pro"
    prec = torch.bfloat16
    queue = launch_thread_safe_queue(checkpoint_path=str(w), device=device, precision=prec, compile=False)
    dec = load_decoder(config_name="modded_dac_vq", checkpoint_path=str(w / "codec.pth"), device=device)
    eng = TTSInferenceEngine(llama_queue=queue, decoder_model=dec, precision=prec, compile=False)

    t1 = time.time()
    paras = [x.strip() for x in text_file.read_text().split("\n\n") if x.strip()]
    if not paras:
        raise SystemExit("empty text")
    refs = [ServeReferenceAudio(audio=r.read_bytes(), text=t) for r, t in zip(ref_paths, a.ref_text)]
    seed = a.seed if a.seed else random.randint(1, 2**31 - 1)
    sr, chunks = None, []
    for i, t in enumerate(paras):
        req = ServeTTSRequest(text=t, references=refs, use_memory_cache="on",  # ref encoded once, cached
                              seed=seed + i, **extra)
        final = None
        for r in eng.inference(req):
            if r.code == "error":
                raise r.error
            if r.code == "final":
                final = r.audio
        sr, wav = final
        if chunks:
            chunks.append(np.zeros(int(a.pause * sr), dtype=np.float32))
        chunks.append(np.asarray(wav, dtype=np.float32).reshape(-1))
    if os.environ.get("FISH_TIMING"):
        print(f"load {t1-t0:.1f}s gen {time.time()-t1:.1f}s audio {sum(len(c) for c in chunks)/sr:.1f}s", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), np.concatenate(chunks), sr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    os._exit(0)  # llama queue thread is non-daemon
