"""Clean a reference clip before cloning. Each enhancer lives in its own venv under enhancers/.

Pipeline: [isolate vocals] -> [enhancer] -> ffmpeg basics (rumble cut, silence trim, loudnorm) -> 24k mono wav.
Usage: python cleanup.py in.m4a out.wav --enhancer dfn [--isolate]
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ENV = Path(__file__).parent / "enhancers"
ENHANCERS = {
    "none": "None (ffmpeg basics only)",
    "dfn": "DeepFilterNet — background noise (fast, recommended)",
    "resemble": "Resemble Enhance — noise + restore muffled audio",
    "voicefixer": "VoiceFixer — rescue heavily degraded / phone audio",
    "clearvoice": "ClearerVoice MossFormer2 — strong denoise",
}

CLEARVOICE_PY = """
import sys
from clearvoice import ClearVoice
cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
cv.write(cv(input_path=sys.argv[1], online_write=False), output_path=sys.argv[2])
"""


def run(*cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed: {r.stderr.strip().splitlines()[-1:]}")


def isolate(src, tmp):
    run(ENV / "demucs/bin/demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", tmp / "demucs", src)
    return next((tmp / "demucs").rglob("vocals.wav"))


def enhance(name, src, tmp):
    out = tmp / f"{name}.wav"
    if name == "dfn":
        run(ENV / "dfn/bin/deepFilter", src, "-o", tmp / "dfn")
        return next((tmp / "dfn").glob("*.wav"))
    if name == "resemble":  # works on directories
        (tmp / "r_in").mkdir()
        shutil.copy(src, tmp / "r_in/in.wav")
        run(ENV / "resemble/bin/resemble-enhance", tmp / "r_in", tmp / "r_out", "--device", "cpu")
        return tmp / "r_out/in.wav"
    if name == "voicefixer":
        run(ENV / "voicefixer/bin/voicefixer", "--infile", src, "--outfile", out, "--mode", "0")
        return out
    if name == "clearvoice":
        run(ENV / "clearvoice/bin/python", "-c", CLEARVOICE_PY, src, out)
        return out
    return src


def cleanup(src, dst, enhancer="dfn", isolate_vocals=False):
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        wav = tmp / "in.wav"  # normalise any input (m4a etc) to wav first
        run("ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", "48000", wav)
        if isolate_vocals:
            wav = isolate(wav, tmp)
        wav = enhance(enhancer, wav, tmp)
        trim = "silenceremove=start_periods=1:start_threshold=-45dB"
        run("ffmpeg", "-y", "-v", "error", "-i", wav, "-af",
            f"highpass=f=80,{trim},areverse,{trim},areverse,loudnorm=I=-20:TP=-2",
            "-ac", "1", "-ar", "24000", dst)
    return dst


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--enhancer", default="dfn", choices=ENHANCERS)
    p.add_argument("--isolate", action="store_true", help="Demucs vocal isolation (music behind voice)")
    a = p.parse_args()
    print(cleanup(a.src, a.dst, a.enhancer, a.isolate))
