"""Rank takes of one script: Whisper word error rate against the script, plus speaker similarity to the reference.
score = similarity - WER (higher is better); without --ref (e.g. dialogue) score = -WER.
Usage: python rank.py --text-file T [--ref REF.wav] take1.wav take2.wav ...   (best path -> stdout, table -> stderr)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

WHISPER = Path(__file__).parent / "engines/whisper"
TAG = re.compile(r"\[[^\]]*\]|<\|[^|]*\|>")  # [sigh], <|speaker:0|>: not spoken as words
_ve = None


def words(t):
    return re.findall(r"[\w']+", TAG.sub(" ", t).lower())


def wer(ref, hyp):
    d = list(range(len(hyp) + 1))  # word-level Levenshtein, one row at a time
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)


def transcribe(wav):
    r = subprocess.run([WHISPER / ".venv/bin/python", WHISPER / "run.py", "--audio", wav], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"whisper failed on {wav}: {r.stderr.strip()[-200:]}")
    return r.stdout


def similarity(a, b):
    global _ve  # Chatterbox's speaker encoder, already in the main venv
    import librosa
    if _ve is None:
        from chatterbox.models.voice_encoder import VoiceEncoder
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        _ve = VoiceEncoder()
        _ve.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox", "ve.safetensors")))
        _ve.eval()
    wavs = [librosa.load(str(f), sr=16000)[0] for f in (a, b)]
    x, y = (_ve.embeds_from_wavs([w], 16000, as_spk=True) for w in wavs)
    return float(x @ y)


def rank(text, takes, ref=None):
    """-> [{path, wer, sim, score}], best first."""
    target, rows = words(text), []
    for t in takes:
        row = {"path": str(t), "wer": wer(target, words(transcribe(t))), "sim": similarity(ref, t) if ref else None}
        row["score"] = (row["sim"] or 0) - row["wer"]
        rows.append(row)
    return sorted(rows, key=lambda r: -r["score"])


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--text-file", required=True)
    p.add_argument("--ref")
    p.add_argument("takes", nargs="+")
    a = p.parse_args()
    rows = rank(Path(a.text_file).read_text(), a.takes, a.ref)
    print(rows[0]["path"])
    for r in rows:
        sim = "-" if r["sim"] is None else f"{r['sim']:.3f}"
        print(f"  score {r['score']:+.3f}  wer {r['wer']:.3f}  sim {sim}  {r['path']}", file=sys.stderr)
