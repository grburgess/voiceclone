"""Auto-tag: insert Fish S2 performance tags ([sigh], [whisper], ...) into a plain script with a local MLX LLM.
Prints the tagged script to stdout. Words are never changed: a paragraph whose words the LLM altered is an error."""
import argparse, os, re, sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
WEIGHTS = str(Path(__file__).parent / "weights/qwen3-4b")

SYSTEM = """You direct voice actors. Insert performance tags in square brackets into the script the user gives you.
Tags go right BEFORE the words they affect (never at the end of a line), or where a sound/pause happens. Use common tags:
[pause] [short pause] [emphasis] [sigh] [inhale] [exhale] [laughing] [chuckle] [whisper] [low voice] [excited]
[surprised] [sad] [angry] [shouting] [volume up] [volume down] [clearing throat]
or a short free-form direction, e.g. [warm and relaxed] or [sarcastic].
Rules:
- Copy every word and punctuation mark exactly. Never add, remove, reorder or change words.
- Keep the same lines: one output line per input line, in order.
- Keep existing [tags] and <|speaker:N|> markers.
- Be sparing: about one tag per sentence at most, only where it improves the delivery.
- Output only the tagged script, nothing else.
Example input:
I can't believe it. We actually won!
So, what now?
Example output:
[surprised] I can't believe it. [excited] We actually won!
[short pause] So, what now?"""

TAG = re.compile(r"\[[^\]]*\]|<\|[^|]*\|>")
PREFIX = re.compile(r"^\s*[\w-]+:\s*")  # dialogue "name:"; kept away from the LLM, which mangles it


def words(t):
    return re.findall(r"[\w']+", TAG.sub(" ", t).lower())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-file", required=True)
    a = p.parse_args()
    paras = [x.strip() for x in Path(a.text_file).read_text().split("\n\n") if x.strip()]
    if not paras:
        raise SystemExit("empty text")
    from mlx_lm import load, generate
    model, tok = load(WEIGHTS)
    out = []
    for i, para in enumerate(paras, 1):
        lines = para.splitlines()
        prefixes = [m[0] if (m := PREFIX.match(l)) else "" for l in lines]
        body = "\n".join(l[len(p):] for l, p in zip(lines, prefixes))
        prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}],
                                         add_generation_prompt=True)
        tagged = [l.strip() for l in generate(model, tok, prompt=prompt, max_tokens=2 * len(tok.encode(body)) + 200)
                  .strip().splitlines() if l.strip()]
        if len(tagged) != len(lines) or words("\n".join(tagged)) != words(body):
            raise SystemExit(f"paragraph {i}: LLM changed the words, not tagging; got: {' / '.join(tagged)[:200]}")
        out.append("\n".join(p + l for p, l in zip(prefixes, tagged)))
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
