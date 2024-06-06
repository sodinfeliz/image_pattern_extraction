VENV             := venv
PYTHON           := $(VENV)/bin/python3
PIP              := $(VENV)/bin/pip
CONFIG_FILE_NAME := user-config

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt torch torchvision torchaudio

build: $(VENV)/bin/activate

configs/$(CONFIG_FILE_NAME).toml: ./configs/config.toml
	@cp ./configs/config.toml ./configs/$(CONFIG_FILE_NAME).toml

run: configs/$(CONFIG_FILE_NAME).toml build
	@$(PYTHON) main.py --config ./configs/$(CONFIG_FILE_NAME).toml

clean:
	rm -rf $(VENV)
	find . -type d -name '__pycache__' -exec rm -rf {} +

.DEFAULT_GOAL := build
.PHONY: all build run clean
