# COFFEE MAPS SURVAILLANCE

## 🔧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourname/coffee-maps-surveillance.git
cd coffee-maps-surveillance
```

### 2. Set up Environment Variables
Copy .env.example → .env and fill in required values (e.g. database connection, API keys).

### Option A: Makefile Installation

If `make` is available in your shell:

```bash
# Create virtual environment and install dependencies
make

# If somehow venv didn't activate - use this command manually
venv\Scripts\activate      # Windows PowerShell / CMD

# run script
python main.py
```

### Option B: Manual Installation

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
venv\Scripts\activate      # Windows PowerShell / CMD

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run a file
python main.py
```