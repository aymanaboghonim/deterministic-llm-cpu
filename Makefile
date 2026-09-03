MODEL_DIR  := models
MODEL_NAME := SmolLM2-135M-Instruct-Q4_K_M.gguf
MODEL_PATH := $(MODEL_DIR)/$(MODEL_NAME)
MODEL_URL  := https://hf-mirror.com/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/$(MODEL_NAME)

.PHONY: install run edit test clean help download

help:
	@echo "Available commands:"
	@echo "  make install   - Create venv and install CPU-only dependencies via uv"
	@echo "  make download  - Download the pre-quantized 105MB SmolLM2 GGUF binary"
	@echo "  make run       - Run deterministic verification in true headless CLI mode"
	@echo "  make edit      - Launch the interactive reactive marimo notebook UI"
	@echo "  make test      - Verify that dependencies contain zero CUDA/Nvidia bloat"
	@echo "  make clean     - Purge caches, temporary files, and local environments"

install:
	uv sync

$(MODEL_PATH):
	@mkdir -p $(MODEL_DIR)
	@echo "[*] Downloading pre-quantized $(MODEL_NAME) (~105 MB)..."
	@curl -f -L -C - --retry 3 "$(MODEL_URL)" -o $(MODEL_PATH)
	@echo "[✓] Download complete: $(MODEL_PATH)"

download: $(MODEL_PATH)

run: $(MODEL_PATH)
	uv run python app.py

edit: $(MODEL_PATH)
	uv run marimo edit app.py

test:
	@echo "[*] Checking for leaked GPU/CUDA dependencies..."
	@uv run pip list | grep -E "nvidia|cuda|triton" && \
		(echo "[!] ERROR: CUDA dependencies detected!" && exit 1) || \
		echo "[✓] Environment is 100% clean CPU-only."

clean:
	rm -rf .venv uv.lock __pycache__ .marimo .pytest_cache models/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete