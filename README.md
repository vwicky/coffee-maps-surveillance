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
python main.py --video-path "your_video_path.mp4"
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
python main.py --video-path "your_video_path.mp4"
```

## Example 
- Full input video: [coffee_shop.mp4](https://drive.google.com/file/d/1QMdTmda82vBEIEZ6PZm2rCs_nnOScm8f/view?usp=drive_link)
- Full processed video output: [processed_video.mp4](https://drive.google.com/file/d/1aLn08MSviUF2ECWHG6OjoDBOTfT8Egmc/view?usp=drive_link)
- MongoDB database view: screenshots

Screenshots of processed frames:

![processed_video-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/b51becb8-d4e7-4e75-815a-c6f5510b820d)


<img width="961" height="579" alt="image" src="https://github.com/user-attachments/assets/53d0ae4c-405a-492b-82c8-3e4f3fed62c3" />
<img width="860" height="577" alt="image_2025-09-18_14-36-01" src="https://github.com/user-attachments/assets/18b864c9-2ecd-428f-9b60-a36bbfa86d89" />

