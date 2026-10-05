import argparse, json, logging, os, sys, random, warnings, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path[:0] = [os.path.join(HERE, "src"), os.path.join(HERE, "src/third_party/Matcha-TTS")]
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

MODEL_DIR = os.path.join(HERE, "weights/Fun-CosyVoice3-0.5B")
WETEXT_DIR = os.path.join(HERE, "weights/wetext")


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
    extra = json.loads(a.extra)

    # Block hidden downloads: wetext normalizer -> local modelscope clone
    import wetext.wetext as _wt
    _wt.snapshot_download = lambda *_, **__: WETEXT_DIR

    import numpy as np, torch, torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice3
    from cosyvoice.utils.common import set_all_random_seed

    torch.set_num_threads(extra.get("threads", os.cpu_count()))
    seed = a.seed or random.randint(1, 2**31 - 1)

    # silence stdout (tqdm/prints) during load + gen
    real_stdout = sys.stdout
    sys.stdout = sys.stderr if extra.get("verbose") else open(os.devnull, "w")
    try:
        cv = CosyVoice3(MODEL_DIR)
        sr = cv.sample_rate
        sys_prompt = extra.get("system_prompt", "You are a helpful assistant.")
        prompt_text = f"{sys_prompt}<|endofprompt|>{a.ref_text}"

        # resample ref to 16k mono into temp wav
        wav, wsr = torchaudio.load(a.ref)
        wav = torchaudio.functional.resample(wav.mean(0, keepdim=True), wsr, 16000)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        torchaudio.save(tmp, wav, 16000)

        paras = [t.strip() for t in open(a.text_file).read().split("\n\n") if t.strip()]
        gap = torch.zeros(1, int(a.pause * sr))
        pieces = []
        for i, para in enumerate(paras):
            set_all_random_seed(seed + i)
            chunks = [o["tts_speech"] for o in cv.inference_zero_shot(
                para, prompt_text, tmp, stream=False,
                speed=float(extra.get("speed", 1.0)),
                text_frontend=bool(extra.get("text_frontend", True)))]
            if pieces:
                pieces.append(gap)
            pieces.append(torch.cat(chunks, dim=1))
        os.unlink(tmp)
        torchaudio.save(a.out, torch.cat(pieces, dim=1), sr)
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr)
        sys.exit(1)
