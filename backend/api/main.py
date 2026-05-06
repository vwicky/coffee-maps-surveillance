import logging
import sys
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import Request
from fastapi.responses import JSONResponse

from api.processor import VideoProcessor
from api.run_local import app
from app_paths import VIDEOS_DIR, ensure_data_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled Exception", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

def main_offline() -> None:
    ensure_data_dirs()
    video_path = VIDEOS_DIR / "cafe_camera_7.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    processor = VideoProcessor()
    result = processor.process_video(str(video_path))
    print("> Processing finished")
    print(f"> Session ID: {result['session_id']}")
    print(f"> Processed video: {result['processed_video']}")
    print(f"> Visitors JSON report: {result['report_json']}")

if __name__ == "__main__":
    print("> all launched")
    main_offline()