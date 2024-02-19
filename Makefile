VENV := env
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt torch torchvision torchaudio

build: $(VENV)/bin/activate

run: build
	@$(PYTHON) main.py

clean:
	rm -rf $(VENV)
	find . -type d -name '__pycache__' -exec rm -rf {} +

.DEFAULT_GOAL := build
.PHONY: all build run clean
