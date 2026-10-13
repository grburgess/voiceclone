#!/usr/bin/env bash
# Install voiceclone: venvs (pinned in requirements/), code checkouts, weights, CLI launcher.
# Usage: ./install.sh [all | main whisper qwen3 voxcpm cosyvoice fish f5 autotag enhancers]   (default: all)
# Needs: macOS arm64, uv, ffmpeg, git, git-lfs, and a network that can reach huggingface.co.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

for t in uv ffmpeg git; do command -v $t >/dev/null || { echo "missing: $t (brew install $t)"; exit 1; }; done
git lfs version >/dev/null 2>&1 || { echo "missing: git-lfs (brew install git-lfs && git lfs install)"; exit 1; }

venv() {  # venv <dir> <python> <requirements-name>
  [ -x "$1/bin/python" ] || uv venv -q -p "$2" "$1"
  VIRTUAL_ENV="$ROOT/$1" uv pip install -q -r "requirements/$3.txt"
}

weights() {  # weights <dest> <git-url> <commit> [lfs-include]
  if [ -d "$1" ] && [ ! -d "$1/.git" ]; then echo "  $1 present (not a git checkout), skipping"; return; fi
  if [ ! -d "$1/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone -q "$2" "$1"
  fi
  git -C "$1" checkout -q "$3"
  git -C "$1" lfs pull ${4:+--include="$4"}
}

code() {  # code <dest> <git-url> <commit>
  [ -d "$1/.git" ] || git clone -q --recurse-submodules "$2" "$1"
  git -C "$1" checkout -q "$3" && git -C "$1" submodule update -q --init --recursive
}

install_main() {
  venv .venv 3.11 main
  # Chatterbox fetches its own weights into ~/.cache/huggingface on first load
  .venv/bin/python -c "from chatterbox.tts import ChatterboxTTS as C; import torch; \
_l=torch.load; torch.load=lambda *a,**k:_l(*a,**{**k,'map_location':'cpu'}); C.from_pretrained(device='cpu')"
}

install_whisper() {
  venv engines/whisper/.venv 3.11 whisper
  weights engines/whisper/weights/base https://huggingface.co/mlx-community/whisper-base-mlx main
}

install_qwen3() {
  venv engines/qwen3/.venv 3.11 qwen3
  weights engines/qwen3/weights/base https://www.modelscope.cn/Qwen/Qwen3-TTS-12Hz-0.6B-Base.git fda1995a3162ad0488393cc366b82ca59c91c08e
}

install_voxcpm() {
  venv engines/voxcpm/.venv 3.11 voxcpm
  weights engines/voxcpm/weights/VoxCPM2 https://huggingface.co/openbmb/VoxCPM2 32279effe8c19989596f05d353d1447f51d9e915
}

install_cosyvoice() {
  code engines/cosyvoice/src https://github.com/FunAudioLLM/CosyVoice 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
  venv engines/cosyvoice/.venv 3.10 cosyvoice
  weights engines/cosyvoice/weights/Fun-CosyVoice3-0.5B https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    29e01c4e8d000f4bcd70751be16fa94bf3d85a18 \
    "llm.pt,flow.pt,hift.pt,campplus.onnx,speech_tokenizer_v3.onnx,CosyVoice-BlankEN/model.safetensors"
  weights engines/cosyvoice/weights/wetext https://www.modelscope.cn/pengzhendong/wetext.git 030bb1febce0bd0168549a03e181cd9c1d70c799
}

install_fish() {
  code engines/fish/src https://github.com/fishaudio/fish-speech 214da3cd841bda85da2496b96cd3c4d7edb1337e
  venv engines/fish/.venv 3.11 fish
  VIRTUAL_ENV="$ROOT/engines/fish/.venv" uv pip install -q --no-deps -e engines/fish/src
  weights engines/fish/weights/s2-pro https://huggingface.co/fishaudio/s2-pro 1de9996b6be38b745688de084d87a5633f714e4e
}

install_f5() {
  venv engines/f5/.venv 3.11 f5
  weights engines/f5/weights/F5-TTS https://huggingface.co/SWivid/F5-TTS 84e5a410d9cead4de2f847e7c9369a6440bdfaca \
    "F5TTS_v1_Base/model_1250000.safetensors,F5TTS_v1_Base/vocab.txt"
  weights engines/f5/weights/vocos https://huggingface.co/charactr/vocos-mel-24khz 0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21
}

install_autotag() {  # local LLM that adds fish performance tags to scripts
  venv engines/autotag/.venv 3.11 autotag
  weights engines/autotag/weights/qwen3-4b https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit \
    50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
}

install_enhancers() {
  venv enhancers/dfn 3.11 dfn
  venv enhancers/resemble 3.10 resemble
  venv enhancers/voicefixer 3.10 voicefixer
  venv enhancers/clearvoice 3.10 clearvoice
  venv enhancers/demucs 3.11 demucs
  # enhancers download their weights on first use; warm them up once with a test tone
  tone="$(mktemp -d)/tone.wav"
  ffmpeg -y -v error -f lavfi -i "sine=frequency=220:duration=3" "$tone"
  for e in dfn resemble voicefixer clearvoice; do
    echo "  warming $e"; .venv/bin/python cleanup.py "$tone" "${tone%.wav}_$e.wav" --enhancer $e >/dev/null || echo "  ! $e warm-up failed"
  done
  .venv/bin/python cleanup.py "$tone" "${tone%.wav}_iso.wav" --enhancer none --isolate >/dev/null || echo "  ! demucs warm-up failed"
}

targets=("$@"); [ ${#targets[@]} -eq 0 ] || [ "${targets[0]}" = all ] && \
  targets=(main whisper qwen3 voxcpm cosyvoice fish f5 autotag enhancers)
[[ " ${targets[*]} " == *" main "* ]] || [ -x .venv/bin/python ] || targets=(main "${targets[@]}")

mkdir -p voices outputs ~/.local/bin
for t in "${targets[@]}"; do echo "==> $t"; "install_$t"; done
ln -sf "$ROOT/bin/voiceclone" ~/.local/bin/voiceclone
echo "Done. Run: voiceclone ui   (ensure ~/.local/bin is on PATH)"
