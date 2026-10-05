#!/usr/bin/env python3
"""Speech-to-text using mlx-whisper for Apple Silicon."""

import sys
import os
from pathlib import Path

# weights vendored locally (weights/base); never download
os.environ["HF_HUB_OFFLINE"] = "1"
WEIGHTS = str(Path(__file__).parent / "weights/base")

try:
    from mlx_whisper import transcribe as mlx_transcribe
    USE_MLX = True
except ImportError:
    USE_MLX = False
    from faster_whisper import WhisperModel


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--audio":
        print("Usage: run.py --audio <path>", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[2]

    # Verify file exists
    if not Path(audio_path).exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    try:
        if USE_MLX:
            # Use mlx-whisper (optimized for Apple Silicon)
            result = mlx_transcribe(
                audio_path,
                path_or_hf_repo=WEIGHTS,
                language="en",
            )
            transcript = result["text"].strip()
        else:
            # Fallback to faster-whisper
            weights_dir = Path(__file__).parent / "weights" / "base"
            if weights_dir.exists():
                model_path = str(weights_dir)
            else:
                model_path = "base"

            model = WhisperModel(model_path, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(audio_path, language="en")
            transcript = " ".join(segment.text for segment in segments)

        # Output only the transcript
        print(transcript)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
