VENV := env

all: run

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip3 install -r requirements.txt
	./$(VENV)/bin/pip3 install torch torchvision torchaudio

build: $(VENV)/bin/activate

run: build
	./$(VENV)/bin/python3 main.py

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete

.PHONY: all build run clean
