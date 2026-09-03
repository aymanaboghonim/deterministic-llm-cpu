[![CI - Deterministic CPU Inference](https://github.com/aymanaboghonim/deterministic-llm-cpu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aymanaboghonim/deterministic-llm-cpu/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)
![Package Manager](https://img.shields.io/badge/uv-Astral-purple)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

# Deterministic LLM Inference on CPU

A minimal, CPU-only script that checks whether greedy autoregressive decoding of a small language model is bit-for-bit reproducible across repeated runs on the same machine.

The script loads [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) from a local Q4_K_M GGUF checkpoint, generates a fixed-length continuation with greedy decoding (`do_sample=False`), and repeats the same generation 100 times. Every run must produce identical token IDs, verified by comparing a SHA-256 digest of the raw token bytes and the decoded text. The process exits `0` when all 100 runs match and `1` on any divergence, so the check runs unattended in CI.

## How it works

All logic lives in `app.py`, a [marimo](https://marimo.io) notebook that also runs as a plain script (`python app.py`). The GGUF checkpoint is stored in 4-bit quantization and de-quantized to `float32` once at load time. The generation prompt is fixed: a chat template with a system prompt and a single user question, and the tokenizer is downloaded from the Hugging Face Hub on first run (downloads default to the `hf-mirror.com` endpoint; an existing `HF_ENDPOINT` is respected).

The verification loop uses the following settings:

| Concern | Setting |
| --- | --- |
| Decoding | `do_sample=False`, `num_beams=1`, `max_new_tokens=25`, `pad_token_id=eos_token_id` |
| Threads | `torch.set_num_threads(2)` |
| Seeds | `PYTHONHASHSEED=0`, `torch.manual_seed(42)` |
| Ops | `torch.use_deterministic_algorithms(True, warn_only=True)` |
| Model state | `model.eval()` inside `torch.inference_mode()` |
| Compute | `float32` on CPU only (CUDA-free wheel index, see below) |
| Check | SHA-256 over the generated token bytes plus decoded text equality, 100/100 runs |

## Scope and limitations

- **Same-machine reproducibility only.** Digests are expected to match across runs on the same CPU with the same library versions. Different CPU architectures or compiler toolchains may round floating-point sums differently, so digests are not expected to match across machines.
- **Reproducibility is not correctness.** A deterministic model repeats its output exactly, including errors or hallucinations. This script says nothing about answer quality.
- **Narrow coverage.** One model, one prompt, and one decoding configuration (greedy, `max_new_tokens=25`, single request). Sampling (`do_sample=True`) is stochastic by design and is not covered.

## Quickstart

Requirements: Linux/macOS (or Windows with WSL2), Python 3.12+, [`uv`](https://docs.astral.sh/uv/), `make`, and `curl`. No GPU or CUDA is required.

```bash
git clone https://github.com/aymanaboghonim/deterministic-llm-cpu.git
cd deterministic-llm-cpu

make install    # create the environment (uv sync, CPU-only torch)
make download   # fetch the GGUF checkpoint into models/ (first run only)
make run        # run the 100-pass verification
```

`make run` prints progress and a summary. Example output from a reference run:

```text
[*] Starting 100 deterministic decoding iterations on CPU...
    [====================] 100/100 (100%) | Digest: eefda54ad8...

=================================================================
VERIFICATION STATUS: PASSED (100% Deterministic)
Iterations Checked : 100 / 100 identical passes
SHA-256 Digest     : eefda54ad8c341911b77331baefcf0e2388167c905e6d7070d4094c1bc04eb20
Canonical Output   : "In concurrent computing, a race condition occurs when two or more threads or processes attempt to access the same data simultaneously, leading to"
=================================================================
```

The digest above was produced on the machine where the reference run was executed; other hardware may produce different values.

## Commands

`app.py` has two modes. As a script it is the headless verification runner shown above, streaming progress to stdout and returning a meaningful exit code. `marimo edit app.py` opens the same notebook in an interactive web UI for experimenting with the prompt and generation parameters.

| Command | What it does |
| --- | --- |
| `make install` | `uv sync` — create the environment from `pyproject.toml` and `uv.lock` |
| `make download` | Fetch the GGUF checkpoint into `models/` with `curl` (resumable) |
| `make run` | `uv run python app.py` — run the 100-pass verification |
| `make edit` | `uv run marimo edit app.py` — open the interactive notebook UI |
| `make test` | Check that no `nvidia-*`, `cuda-*`, or `triton` packages are installed |
| `make clean` | Remove `.venv`, `uv.lock`, `models/`, and caches |

`make test` passes when the environment contains no GPU packages:

```text
[*] Checking for leaked GPU/CUDA dependencies...
[✓] Environment is 100% clean CPU-only.
```

## CPU-only environment

`pyproject.toml` pins PyTorch to the CPU wheel index so `uv sync` does not pull CUDA libraries:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request to `main` on `ubuntu-latest`. It installs Python 3.12 and the CPU-only dependencies, runs the purity audit (`make test`), caches the GGUF checkpoint, then executes the 100-pass verification (`make run`). The workflow pins `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=2`, and `MKL_NUM_THREADS=2`.

## Repository layout

```text
.
├── .github/workflows/ci.yml   # CI: install, purity audit, 100-pass verification
├── app.py                     # marimo notebook / headless verification script
├── Makefile                   # command entry points
├── pyproject.toml             # dependencies and CPU-only torch index
├── uv.lock                    # locked dependency versions
├── .gitignore
├── LICENSE                    # MIT
├── README.md
└── models/                    # GGUF checkpoint, created by make download (gitignored)
```

## License

[MIT](LICENSE)