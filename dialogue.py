"""Dialogue scripts for Fish S2: lines like "morgan: Hello." -> "<|speaker:0|>Hello.", one reference voice per speaker.

A prefix counts only if voices/<name>.wav exists; lines without one continue the previous speaker.
Usage: python dialogue.py script.txt voices/   (rewrites script.txt in place, prints speaker voice names in order)
"""
import re
import sys
from pathlib import Path

LINE = re.compile(r"^(\s*)([\w-]+):\s*(.*)$")


def parse(text, voices):
    """-> (text with <|speaker:N|> markers, [voice name for speaker 0, 1, ...]); names empty if not a dialogue."""
    names, lines = [], []
    for line in text.splitlines():
        m = LINE.match(line)
        if m and (Path(voices) / f"{m[2].lower()}.wav").exists():
            name = m[2].lower()
            if name not in names:
                names.append(name)
            line = f"{m[1]}<|speaker:{names.index(name)}|>{m[3]}"
        lines.append(line)
    return ("\n".join(lines), names) if names else (text, [])


if __name__ == "__main__":
    script = Path(sys.argv[1])
    text, names = parse(script.read_text(), sys.argv[2])
    script.write_text(text)
    print("\n".join(names))
