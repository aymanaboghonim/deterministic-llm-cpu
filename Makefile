.PHONY: install run edit test clean help

help:
	@echo "Available commands:"
	@echo "  make install  - Create venv and install CPU-only dependencies via uv"
	@echo "  make run      - Run deterministic verification in headless CLI mode"
	@echo "  make edit     - Launch the interactive reactive marimo notebook UI"
	@echo "  make test     - Verify that dependencies contain zero CUDA/Nvidia bloat"
	@echo "  make clean    - Purge caches, temporary files, and local environments"

install:
	uv sync

run:
	uv run marimo run app.py

edit:
	uv run marimo edit app.py

test:
	@echo "[*] Checking for leaked GPU/CUDA dependencies..."
	@uv run pip list | grep -E "nvidia|cuda|triton" && \
		(echo "[!] ERROR: CUDA dependencies detected!" && exit 1) || \
		echo "[✓] Environment is 100% clean CPU-only."

clean:
	rm -rf .venv uv.lock __pycache__ .marimo .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete