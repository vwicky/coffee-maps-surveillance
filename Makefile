PYTHON = python
VENV = venv
PIP = $(VENV)\Scripts\python -m pip
PY = $(VENV)\Scripts\python

# Default target
all: venv install

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV)

# Install requirements
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Run a Python script inside the venv
run-script: venv
	@set /p script="Enter the Python script you want to run (e.g. main.py): " && $(PY) %script%

# Open an interactive shell with the venv activated
shell: venv
	@echo "Entering venv shell. Type 'exit' to leave."
	@cmd /k "$(VENV)\Scripts\activate"

# Clean up venv
clean:
	rmdir /s /q $(VENV)
