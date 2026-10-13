"""Local web UI for Chatterbox + other TTS engines (engines/<name>/run.py, own venvs). Run: .venv/bin/python app.py  -> http://127.0.0.1:7860"""
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

from cleanup import ENHANCERS, cleanup
from dialogue import parse as parse_dialogue
from rank import rank

device = "mps" if torch.backends.mps.is_available() else "cpu"
# checkpoints saved on CUDA; remap on load
_load = torch.load
torch.load = lambda *x, **k: _load(*x, **{**k, "map_location": device})
_model = None


def chatterbox():
    global _model  # lazy: only pay the load when Chatterbox is used
    if _model is None:
        _model = ChatterboxTTS.from_pretrained(device=device)
    return _model


ENGINES = Path(__file__).parent / "engines"
NEEDS_TRANSCRIPT = {"cosyvoice", "fish", "f5"}
MODELS = {  # key -> label (license); non-chatterbox run via engines/<key>/run.py
    "chatterbox": "Chatterbox — MIT, fast, weak accent",
    "qwen3": "Qwen3-TTS 0.6B — Apache-2.0, ~15s/sentence",
    "voxcpm": "VoxCPM2 — Apache-2.0, ~25s/sentence",
    "cosyvoice": "CosyVoice 3 — Apache-2.0, CPU, ~30s/sentence",
    "fish": "Fish S2 Pro — NON-COMMERCIAL, ~70s/sentence",
    "f5": "F5-TTS — NON-COMMERCIAL, CPU, ~60s/sentence",
}

VOICES = Path(__file__).parent / "voices"
OUTPUTS = Path(__file__).parent / "outputs"
DEFAULT = "(default voice)"
MEMOS = Path.home() / "Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings"


def memo_choices():
    files = sorted(MEMOS.glob("*.m4a"), key=lambda f: f.stat().st_mtime, reverse=True)[:20]
    return [(datetime.fromtimestamp(f.stat().st_mtime).strftime("%b %d %H:%M"), str(f)) for f in files]


def voice_choices():
    return [DEFAULT] + sorted(f.name for f in VOICES.iterdir() if f.suffix.lower() in {".wav", ".m4a", ".mp3", ".flac"})


def transcript(voice):
    """Cached whisper transcript beside the voice file (voices/<name>.txt, user-editable)."""
    if voice == DEFAULT:
        return ""
    txt = (VOICES / voice).with_suffix(".txt")
    if not txt.exists():
        r = subprocess.run([ENGINES / "whisper/.venv/bin/python", ENGINES / "whisper/run.py",
                            "--audio", VOICES / voice], capture_output=True, text=True)
        if r.returncode or not r.stdout.strip():
            return ""  # leave uncached so it retries next time
        txt.write_text(r.stdout.strip())
    return txt.read_text()


def save_transcript(voice, text):
    if voice != DEFAULT and text.strip():
        (VOICES / voice).with_suffix(".txt").write_text(text.strip())


def clip_loaded(clip):
    if not clip:
        return gr.update(), gr.update()
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", clip],
                               capture_output=True, text=True).stdout)
    return gr.update(maximum=dur, value=0), gr.update(maximum=dur, value=dur)


def save_voice(clip, name, clean, enhancer, isolate, start, end):
    if not clip or not name.strip():
        raise gr.Error("Need a clip and a name")
    if end <= start:
        raise gr.Error("End must be after start")
    trimmed = Path(tempfile.mkdtemp()) / "trimmed.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-to", str(end), "-i", clip,
                    "-ac", "1", "-ar", "48000", str(trimmed)], check=True)
    clip = str(trimmed)
    dst = VOICES / f"{name.strip()}.wav"
    # ffmpeg handles m4a/mp3/etc; torchaudio's soundfile backend can't read m4a
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", clip, "-ac", "1", "-ar", "24000",
                    str(VOICES / f"{name.strip()}_raw.wav" if clean else dst)], check=True)
    if clean:
        try:
            cleanup(clip, dst, enhancer, isolate)
        except RuntimeError as e:
            raise gr.Error(str(e))
    dst.with_suffix(".txt").unlink(missing_ok=True)  # stale transcript from an overwritten voice
    transcript(dst.name)
    return gr.update(choices=voice_choices(), value=f"{name.strip()}.wav"), str(dst)


TAGS_HELP = """**Fish S2 directions** — put tags right where they apply:
`[pause]` `[short pause]` `[emphasis]` `[sigh]` `[inhale]` `[exhale]` `[laughing]` `[chuckle]` `[whisper]` `[low voice]`
`[excited]` `[surprised]` `[sad]` `[angry]` `[shouting]` `[volume up]` `[volume down]` `[clearing throat]`, or free-form:
`[warm and relaxed, like talking to an old friend]`.

**Dialogue** — one pass, a voice per speaker: start lines with a voice name, e.g. `morgan: Hello.` / `sam: Hi!`
(the Voice dropdown is ignored; every speaker needs a transcript)."""


def autotag(text):
    if not text.strip():
        raise gr.Error("Script is empty")
    tf = Path(tempfile.mkdtemp()) / "script.txt"
    tf.write_text(text)
    r = subprocess.run([ENGINES / "autotag/.venv/bin/python", ENGINES / "autotag/run.py", "--text-file", tf],
                       capture_output=True, text=True)
    if r.returncode:
        raise gr.Error(f"auto-tag failed: {(r.stderr.strip().splitlines() or ['no output'])[-1]}")
    return r.stdout.strip()


def generate(engine, text, voice, ref_text, exaggeration, cfg, temperature, pause, seed, name, takes):
    if not text.strip():
        raise gr.Error("Script is empty")
    path = OUTPUTS / f"{datetime.now():%Y%m%d-%H%M%S}_{engine}_{name.strip() or 'clip'}.wav"
    takes, seed = int(takes), int(seed)
    paths = [path] if takes == 1 else [path.with_name(f"{path.stem}_take{k}.wav") for k in range(takes)]
    speakers = []
    if engine == "fish":
        text, speakers = parse_dialogue(text, VOICES)
    if engine != "chatterbox":
        if speakers:
            refs = [(VOICES / f"{v}.wav", transcript(f"{v}.wav")) for v in speakers]
            if missing := [v for v, (_, t) in zip(speakers, refs) if not t]:
                raise gr.Error(f"no transcript for: {', '.join(missing)}")
        else:
            if voice == DEFAULT:
                raise gr.Error(f"{engine} needs a reference voice")
            if engine in NEEDS_TRANSCRIPT and not ref_text.strip():
                raise gr.Error(f"{engine} needs a voice transcript — fill in the transcript box")
            save_transcript(voice, ref_text)
            refs = [(VOICES / voice, ref_text.strip())]
        tf = Path(tempfile.mkdtemp()) / "script.txt"
        tf.write_text(text)
        for k, p in enumerate(paths):
            cmd = [ENGINES / engine / ".venv/bin/python", ENGINES / engine / "run.py", "--text-file", tf,
                   "--out", p, "--pause", pause, "--seed", seed + k if seed else 0]
            for ref, rt in refs:
                cmd += ["--ref", ref] + (["--ref-text", rt] if rt else [])
            r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=ENGINES / engine)
            if r.returncode:
                raise gr.Error(f"{engine} failed: {(r.stderr.strip().splitlines() or ['no output'])[-1]}")
    else:
        model = chatterbox()
        prompt = None if voice == DEFAULT else str(VOICES / voice)
        paras = [s.strip() for s in text.split("\n\n") if s.strip()]
        gap = torch.zeros(1, int(model.sr * pause))
        for k, p in enumerate(paths):
            if seed:
                torch.manual_seed(seed + k)
            chunks = []
            for para in paras:
                chunks += [model.generate(para, audio_prompt_path=prompt, exaggeration=exaggeration,
                                          cfg_weight=cfg, temperature=temperature).cpu(), gap]
            ta.save(str(p), torch.cat(chunks[:-1], dim=1), model.sr)
    if takes == 1:
        return str(path), f"Saved: {path}"
    ref = None if speakers or voice == DEFAULT else VOICES / voice  # no single reference to compare a dialogue to
    rows = rank(text, paths, ref)
    Path(rows[0]["path"]).rename(path)  # best take gets the plain name; the others keep _takeK
    table = "\n".join(f"| {'**best**' if i == 0 else Path(r['path']).name} | {r['score']:+.3f} | {r['wer']:.3f} | "
                      f"{'-' if r['sim'] is None else format(r['sim'], '.3f')} |" for i, r in enumerate(rows))
    return str(path), f"Saved best of {takes}: {path}\n\n| take | score | WER | speaker sim |\n|---|---|---|---|\n{table}"


with gr.Blocks(title="Chatterbox") as demo:
    gr.Markdown("## Voice Cloning TTS\nNon-Chatterbox models reload each run (times include load).\nBlank line = new paragraph (generated separately, joined with a pause).")
    with gr.Row():
        with gr.Column():
            engine = gr.Dropdown([(v, k) for k, v in MODELS.items()], value="chatterbox", label="Model")
            text = gr.Textbox(label="Script", lines=10)
            with gr.Accordion("Directing (Fish S2): tags, dialogue, auto-tag", open=False, visible=False) as fish_help:
                gr.Markdown(TAGS_HELP)
                gr.Button("✨ Auto-tag script (local LLM, ~10 s)", size="sm").click(autotag, text, text)
            voice = gr.Dropdown(voice_choices(), value="me.wav", label="Voice (from voices/)")
            ref_text = gr.Textbox(transcript("me.wav"), label="Voice transcript (auto by Whisper; fix mistakes). Required: CosyVoice/Fish/F5. Blank = weaker clone for Qwen3/VoxCPM", lines=2)
            voice.change(transcript, voice, ref_text)
            with gr.Accordion("Add a new voice", open=False):
                memo = gr.Dropdown(memo_choices(), label="Pick a Voice Memo (newest first)")
                gr.Button("↻ Refresh memos", size="sm").click(lambda: gr.update(choices=memo_choices()), None, memo)
                clip = gr.Audio(label="Clip (5-15s, upload or record)", type="filepath")
                memo.change(lambda m: m, memo, clip)
                vname = gr.Textbox(label="Voice name")
                with gr.Row():
                    start = gr.Slider(0, 60, 0, step=0.1, label="Keep from (s)")
                    end = gr.Slider(0, 60, 60, step=0.1, label="Keep to (s)")
                kept = gr.Markdown()
                clip.change(clip_loaded, clip, [start, end])
                for sl in (start, end):
                    sl.change(lambda a, b: f"Keeping **{max(b - a, 0):.1f}s** (aim for 10–20s)", [start, end], kept)
                clean = gr.Checkbox(True, label="Clean up audio (keeps original as <name>_raw.wav)")
                enhancer = gr.Dropdown([(v, k) for k, v in ENHANCERS.items()], value="dfn", label="Enhancer")
                isolate = gr.Checkbox(False, label="Isolate voice from music (Demucs, slower)")
                saved_voice = gr.Audio(label="Saved voice (cleaned) — listen here", type="filepath", interactive=False)
                gr.Button("Save to voices/").click(save_voice, [clip, vname, clean, enhancer, isolate, start, end], [voice, saved_voice])
        with gr.Column():
            with gr.Group() as cb_opts:
                exaggeration = gr.Slider(0.25, 2.0, 0.5, step=0.05, label="Exaggeration (calm → dramatic) — Chatterbox")
                cfg = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="CFG / pace (lower = slower) — Chatterbox")
                temperature = gr.Slider(0.05, 5.0, 0.8, step=0.05, label="Temperature (variation) — Chatterbox")
            engine.change(lambda e: gr.update(visible=e == "chatterbox"), engine, cb_opts)
            engine.change(lambda e: gr.update(visible=e == "fish"), engine, fish_help)
            pause = gr.Slider(0.0, 2.0, 0.4, step=0.1, label="Pause between paragraphs (s)")
            seed = gr.Number(0, label="Seed (0 = random)", precision=0)
            takes = gr.Slider(1, 5, 1, step=1, label="Takes — generate N, keep the best (Whisper accuracy + voice match); N× slower")
            name = gr.Textbox(label="Output name (optional)")
            btn = gr.Button("Generate", variant="primary")
            out = gr.Audio(label="Output", type="filepath")
            saved = gr.Markdown()
    btn.click(generate, [engine, text, voice, ref_text, exaggeration, cfg, temperature, pause, seed, name, takes], [out, saved])

demo.launch(allowed_paths=[str(MEMOS)])
