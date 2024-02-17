VENV := env
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt torch torchvision torchaudio

build: $(VENV)/bin/activate

run: build
	@$(PYTHON) main.py

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete

.DEFAULT_GOAL := build
.PHONY: all build run clean
