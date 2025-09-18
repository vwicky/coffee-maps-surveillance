# COFFEE MAPS SURVAILLANCE

## Overview

CoffeeMaps Surveillance is a computer vision system for tracking and recognizing visitors in public venues using surveillance cameras.
The system takes a video as input and produces multiple outputs that help monitor visitor behavior and demographics.

## Features
- Detects and tracks people in video streams.
- Assigns a unique ID to each detected person.
- Outputs a processed video with bounding boxes and IDs for each person.
- Logs entry and exit times for each visitor.
- Generates a CSV table with the duration of each visitor’s stay.
- Stores snapshots of visitors for further analysis.
- Populates a MongoDB database with classifications, including gender, age, and race.

## System Requirements
- Python 3.11
- Libraries:
    ultralytics==8.3.0
    opencv-python==4.10.0.84
    pyautogui
    pymongo
    numpy
    Other dependencies listed in requirements.txt

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

## Notes
- Ensure MongoDB is running and accessible for demographic classification storage.
- The system currently supports pre-recorded video files; live camera support can be added.
- Press "q" to stop video preview during processing (if GUI is enabled).
