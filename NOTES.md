# NOTES

Running log of decisions, numbers and dead ends. Newest at the bottom of each section.

## Machine (2026-09-10)

- 8 × RTX A6000 48 GB, driver 550.90.07 → CUDA 12.4 at most. GPU 0 is occupied by a
  transcriptions API (gunicorn, 18 GB) and `ollama serve` is resident: **train on GPUs 1–7 only**.
- 126 cores, 679 GB RAM, /dev/shm 340 GB, ~198 GB free on /dev/vda1.
- No sudo. `uv` 0.10.9, `gh` and `hf` CLIs already authenticated.
- Torch: driver 550 cannot run cu130 wheels (needs ≥ 580), so torch is pinned to the cu126 index.

## Environment

- Python 3.12 via uv; deps in `pyproject.toml`.
