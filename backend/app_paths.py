from __future__ import annotations

from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"

VIDEOS_DIR = DATA_ROOT / "videos"
SNAPSHOTS_DIR = DATA_ROOT / "snapshots"
LOGS_DIR = DATA_ROOT / "logs"
DATASET_RAW_DIR = DATA_ROOT / "dataset_raw"
MERGED_GALLERY_DIR = DATA_ROOT / "merged_gallery"
UPLOADS_DIR = DATA_ROOT / "uploads"
ZONES_JSON_PATH = DATA_ROOT / "zones.json"


def ensure_data_dirs() -> None:
    for path in (
        DATA_ROOT,
        VIDEOS_DIR,
        SNAPSHOTS_DIR,
        LOGS_DIR,
        DATASET_RAW_DIR,
        MERGED_GALLERY_DIR,
        UPLOADS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
