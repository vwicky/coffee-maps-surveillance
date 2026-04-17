import cv2
import json
import numpy as np
import sys
from functools import lru_cache
from pathlib import Path
import time
import os
from app_paths import DATA_ROOT

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app_paths import UPLOADS_DIR, ZONES_JSON_PATH, ensure_data_dirs
from detection.staff_detector import StaffDetector

app = FastAPI(title="Cafe Monitor API")

# Налаштування CORS, щоб React міг достукатися до бекенду
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_data_dirs()
UPLOAD_DIR = UPLOADS_DIR


@lru_cache(maxsize=1)
def get_model():
    """Load YOLO only when the video stream endpoint is actually used."""
    from ultralytics import YOLO

    return YOLO("yolov8l.pt")


def _normalize_zone_points(points):
    if not isinstance(points, list) or len(points) < 3:
        return None

    zone = np.array(points, dtype=np.int32)
    if zone.ndim != 2 or zone.shape[1] != 2:
        return None

    return zone


def _extract_barista_zone(payload):
    preferred_keys = ("barista_zone", "barista", "staff_zone", "staff", "session_1")

    if isinstance(payload, dict):
        for key in preferred_keys:
            zone = _normalize_zone_points(payload.get(key))
            if zone is not None:
                return zone

        for value in payload.values():
            if isinstance(value, dict):
                for key in preferred_keys:
                    zone = _normalize_zone_points(value.get(key))
                    if zone is not None:
                        return zone

    return None


def _load_barista_zone():
    try:
        with ZONES_JSON_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None

    return _extract_barista_zone(payload)


def _draw_barista_zone(frame, barista_zone):
    if barista_zone is None:
        return

    cv2.polylines(
        frame,
        [barista_zone.reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def _crop_person(frame, bbox):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def _is_point_inside_zone(zone, point):
    if zone is None:
        return False

    return cv2.pointPolygonTest(zone, point, False) >= 0

def generate_yolo_frames(video_path):
    model = get_model() 
    
    # 1. ЗАВАНТАЖУЄМО ЗОНИ
    barista_zone = None
    zones_path = os.path.join(DATA_ROOT, "zones.json") # Або "../zones.json" залежно від вашої структури
    if os.path.exists(zones_path):
        with open(zones_path, "r") as f:
            data = json.load(f)
            # Переконайтеся, що ключ відповідає вашому json (напр. 'barista_zone' чи 'session_1')
            if "barista_zone" in data: 
                barista_zone = np.array(data["barista_zone"], np.int32)
                
    entry_times = {}

    # 2. МАГІЯ ТУТ: Передаємо шлях до відео прямо в модель.
    # YOLO сама відкриє відео, сама розіб'є на кадри і, головне, САМА ОЧИСТИТЬ ТРЕКЕР!
    results_generator = model.track(
        source=video_path, 
        stream=True, 
        persist=True, 
        classes=[0], 
        conf=0.4, 
        tracker="botsort.yaml", 
        verbose=False
    )

    # Йдемо по результатах генератора
    for r in results_generator:
        # r.orig_img - це чистий оригінальний кадр (як frame раніше)
        annotated_frame = r.orig_img.copy()
        current_time = time.time()

        # Малюємо червону зону баристи
        if barista_zone is not None and len(barista_zone) > 0:
            cv2.polylines(annotated_frame, [barista_zone], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(annotated_frame, "Barista Zone", tuple(barista_zone[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                tid = track_ids[i] if i < len(track_ids) else None
                
                # --- ЛОГІКА ЗОН (ігноруємо людей в зоні баристи) ---
                center_x = int((x1 + x2) / 2)
                bottom_y = int(y2)
                
                if barista_zone is not None and len(barista_zone) > 0:
                    if cv2.pointPolygonTest(barista_zone, (center_x, bottom_y), False) >= 0:
                        continue 

                # --- ЛОГІКА СТАФФУ (Розкоментуйте, якщо імпортували staff_detector) ---
                # crop = r.orig_img[y1:y2, x1:x2]
                # if is_staff(crop):
                #     continue
                
                # --- МАЛЮВАННЯ КЛІЄНТІВ ---
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if tid is not None:
                    tid = int(tid)
                    if tid not in entry_times:
                        entry_times[tid] = current_time
                    time_in_frame = int(current_time - entry_times[tid])
                    label = f"Customer {tid} - {time_in_frame}s"
                    bg_color = (0, 255, 0) 
                else:
                    label = "Tracking..."
                    bg_color = (0, 165, 255) 
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw, y1), bg_color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Перетворюємо в картинку для відправки в браузер
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    file_path = UPLOAD_DIR / video.filename
    with file_path.open("wb") as buffer:
        buffer.write(await video.read())
    
    # Повертаємо посилання на майбутній стрім
    return {
        "status": "uploaded",
        "filename": video.filename,
        "stream_url": f"http://localhost:8000/video_feed/{video.filename}"
    }

@app.get("/video_feed/{filename}")
async def video_feed(filename: str):
    video_path = UPLOAD_DIR / filename
    return StreamingResponse(
        generate_yolo_frames(str(video_path)),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)