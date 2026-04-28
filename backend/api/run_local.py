import cv2
import json
import logging
import numpy as np
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from pymongo import MongoClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app_paths import (
    CALIBRATION_JSON_PATH,
    DATA_ROOT,
    FLOOR_PLAN_PATH,
    SNAPSHOTS_DIR,
    THUMBNAILS_DIR,
    UPLOADS_DIR,
    ZONES_JSON_PATH,
    ensure_data_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Cafe Monitor API")

# Налаштування CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_data_dirs()
UPLOAD_DIR = UPLOADS_DIR
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")

mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["cafe_analytics"]
customers_collection = mongo_db["customers"]
mongo_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mongo-writer")
current_occupancy: dict[str, list[dict[str, int]]] = {}


class Point(BaseModel):
    x: float
    y: float


class CalibrationPayload(BaseModel):
    src_pts: list[Point]
    dst_pts: list[Point]


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled Exception", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

def _build_unique_filename(original_name: str) -> str:
    source_path = Path(original_name or "upload.mp4")
    stem = source_path.stem or "upload"
    suffix = source_path.suffix or ".mp4"
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem).strip("._")
    if not safe_stem:
        safe_stem = "upload"
    return f"{safe_stem}_{int(time.time() * 1000)}_{uuid4().hex[:8]}{suffix}"


def _save_video_thumbnail(video_path: Path) -> str | None:
    ensure_data_dirs()
    thumb_name = f"{video_path.stem}_thumb.jpg"
    thumb_path = THUMBNAILS_DIR / thumb_name

    capture = cv2.VideoCapture(os.fspath(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        return None

    if not cv2.imwrite(os.fspath(thumb_path), frame):
        return None

    return thumb_name


def estimate_age_gender(cropped_image_numpy_array: np.ndarray) -> tuple[int, str]:
    """Estimate age and gender for a customer crop.

    Uses the existing DeepFace-based stack when available. If inference fails,
    falls back to a clear placeholder so the user can replace it with a custom model.
    """
    try:
        from deepface import DeepFace

        results = DeepFace.analyze(
            img_path=cropped_image_numpy_array,
            actions=["age", "gender"],
            enforce_detection=False,
            silent=True,
        )
        face_data = results[0] if isinstance(results, list) else results
        age = int(face_data.get("age", 25))
        gender_scores = face_data.get("gender", {}) or {}
        man_score = float(gender_scores.get("Man", 0.0))
        woman_score = float(gender_scores.get("Woman", 0.0))
        gender = "Female" if woman_score >= man_score else "Male"
        return age, gender
    except Exception:
        # Placeholder fallback until a dedicated production model is plugged in.
        return 25, "Female"


def _safe_crop_person(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
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


def _save_customer_snapshot(
    crop: np.ndarray,
    camera_id: int | None,
    track_id: int,
) -> str | None:
    ensure_data_dirs()
    camera_label = camera_id if camera_id is not None else "unknown"
    snapshot_filename = f"camera_{camera_label}_person_{track_id}.jpg"
    snapshot_path = SNAPSHOTS_DIR / snapshot_filename
    if not cv2.imwrite(os.fspath(snapshot_path), crop):
        return None
    return snapshot_filename


def _insert_customer_record(document: dict) -> None:
    try:
        customers_collection.insert_one(document)
    except Exception:
        pass


def _normalize_gender_label(gender: object) -> str:
    value = str(gender or "").strip().lower()
    if value in {"male", "man", "m"}:
        return "Male"
    if value in {"female", "woman", "f"}:
        return "Female"
    return "Unknown / other"


def _age_bucket_from_value(age: object) -> str:
    try:
        age_value = int(float(age))
    except (TypeError, ValueError):
        age_value = -1

    if age_value < 0:
        return "Unknown"
    if age_value <= 17:
        return "0-17"
    if age_value <= 25:
        return "18-25"
    if age_value <= 35:
        return "26-35"
    if age_value <= 45:
        return "36-45"
    if age_value <= 55:
        return "46-55"
    return "56+"


def _zone_key_for_filename(filename: str) -> str:
    return f"file_{filename}_barista_zone"


def _calibration_key_for_filename(filename: str) -> str:
    return f"file_{filename}_calibration"


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def _write_json_file(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _load_zone_for_filename(filename: str):
    data = _load_json_file(ZONES_JSON_PATH)
    return data.get(_zone_key_for_filename(filename))


def _load_calibration_for_filename(filename: str):
    data = _load_json_file(CALIBRATION_JSON_PATH)
    return data.get(_calibration_key_for_filename(filename))


def _merge_projected_points(
    occupancy_by_camera: dict[str, list[dict[str, int]]],
    distance_threshold: float = 50.0,
) -> list[dict[str, int]]:
    clusters: list[dict[str, object]] = []

    for camera_filename, points in occupancy_by_camera.items():
        for point in points:
            x = point.get("x")
            y = point.get("y")
            if x is None or y is None:
                continue

            best_cluster: dict[str, object] | None = None
            best_distance = float("inf")

            for cluster in clusters:
                cluster_cameras = cluster["cameras"]
                if camera_filename in cluster_cameras:
                    continue

                distance = float(np.hypot(float(cluster["x"]) - x, float(cluster["y"]) - y))
                if distance <= distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_cluster = cluster

            if best_cluster is None:
                clusters.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "count": 1,
                        "cameras": {camera_filename},
                    }
                )
                continue

            count = int(best_cluster["count"])
            best_cluster["x"] = (float(best_cluster["x"]) * count + x) / (count + 1)
            best_cluster["y"] = (float(best_cluster["y"]) * count + y) / (count + 1)
            best_cluster["count"] = count + 1
            best_cluster["cameras"].add(camera_filename)

    return [
        {"x": int(round(float(cluster["x"]))), "y": int(round(float(cluster["y"])))}
        for cluster in clusters
    ]


def generate_yolo_frames(video_path, filename, camera_id=None):
    from ultralytics import YOLO

    logger.info("Starting stream for %s", filename)
    print(f"▶️ STREAM STARTED: {filename}")
    model = YOLO("yolov8s.pt")
    processed_customer_ids = set()
    homography_status_logged = False
    logger.info("Opening video source: %s", video_path)
    capture = cv2.VideoCapture(os.fspath(video_path))
    if not capture.isOpened():
        logger.critical("Failed to open video source: %s", video_path)
    else:
        logger.info("Video source opened successfully: %s", video_path)
    capture.release()
    plan_img = cv2.imread(os.fspath(FLOOR_PLAN_PATH)) if FLOOR_PLAN_PATH.exists() else None
    if plan_img is not None:
        floor_h, floor_w = plan_img.shape[:2]
    else:
        floor_h, floor_w = 1000, 1000

    results_generator = model.track(
        source=video_path,
        stream=True,
        persist=True,
        classes=[0],
        conf=0.25,
        vid_stride=3,
        tracker="botsort.yaml",
        verbose=False
    )

    try:
        for r in results_generator:
            annotated_frame = r.orig_img.copy()
            projected_points: list[dict[str, int]] = []
            try:
                current_time = time.time()
                height, width = annotated_frame.shape[:2]
                raw_zone = _load_zone_for_filename(filename)
                calibration = _load_calibration_for_filename(filename)
                homography_matrix = None

                barista_zone = None
                if raw_zone and len(raw_zone) >= 3:
                    pts = [[int(p["x"] * width), int(p["y"] * height)] for p in raw_zone]
                    barista_zone = np.array(pts, np.int32)

                if calibration:
                    raw_src_pts = calibration.get("src_pts", [])
                    raw_dst_pts = calibration.get("dst_pts", [])
                    if len(raw_src_pts) >= 4 and len(raw_dst_pts) >= 4:
                        try:
                            src_pts = np.array(
                                [[float(p["x"]) * width, float(p["y"]) * height] for p in raw_src_pts[:4]],
                                dtype=np.float32,
                            )
                            dst_pts = np.array(
                                [[float(p["x"]) * floor_w, float(p["y"]) * floor_h] for p in raw_dst_pts[:4]],
                                dtype=np.float32,
                            )
                            homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
                            if homography_matrix is None:
                                logger.warning("Homography calculation returned None for %s", filename)
                            elif not homography_status_logged:
                                logger.info("Loaded homography matrix for %s", filename)
                                homography_status_logged = True
                        except Exception as exc:
                            homography_matrix = None
                            logger.warning("Homography calculation failed for %s: %s", filename, exc)
                    elif not homography_status_logged:
                        logger.warning("No homography points found for %s", filename)
                        homography_status_logged = True
                elif not homography_status_logged:
                    logger.warning("No homography points found for %s", filename)
                    homography_status_logged = True

                if barista_zone is not None:
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [barista_zone], (0, 0, 255))
                    cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                    cv2.polylines(annotated_frame, [barista_zone], isClosed=True, color=(0, 0, 255), thickness=2)
                    cv2.putText(annotated_frame, "Barista Zone", tuple(barista_zone[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)

                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        tid = track_ids[i] if i < len(track_ids) else None
                        center_x = int((x1 + x2) / 2)
                        bottom_y = int(y2)

                        is_staff = False
                        if barista_zone is not None:
                            is_staff = cv2.pointPolygonTest(barista_zone, (center_x, bottom_y), False) >= 0

                        if is_staff:
                            label = "Staff"
                            bg_color = (0, 165, 255)
                        else:
                            label = "Customer"
                            bg_color = (0, 255, 0)

                        map_x = None
                        map_y = None
                        if not is_staff and homography_matrix is not None:
                            try:
                                pt = np.array([[[center_x, bottom_y]]], dtype=np.float32)
                                mapped_pt = cv2.perspectiveTransform(pt, homography_matrix)[0][0]
                                if np.isfinite(mapped_pt).all():
                                    map_x = int(mapped_pt[0])
                                    map_y = int(mapped_pt[1])
                            except Exception as exc:
                                map_x = None
                                map_y = None
                                logger.warning(
                                    "Perspective transform failed for %s track %s: %s",
                                    filename,
                                    tid,
                                    exc,
                                )

                        if map_x is not None and map_y is not None:
                            projected_points.append({"x": map_x, "y": map_y})

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bg_color, 2)

                        if not is_staff and tid is not None:
                            tid = int(tid)
                            if tid not in processed_customer_ids:
                                crop = _safe_crop_person(r.orig_img, (x1, y1, x2, y2))
                                if crop is not None:
                                    estimated_age, estimated_gender = estimate_age_gender(crop)
                                    snapshot_filename = _save_customer_snapshot(crop, camera_id, tid)
                                    if snapshot_filename is not None:
                                        processed_customer_ids.add(tid)
                                        mongo_executor.submit(
                                            _insert_customer_record,
                                            {
                                                "camera_id": camera_id,
                                                "track_id": tid,
                                                "entry_time": current_time,
                                                "age": estimated_age,
                                                "gender": estimated_gender,
                                                "map_x": map_x,
                                                "map_y": map_y,
                                                "snapshot_filename": snapshot_filename,
                                            },
                                        )

                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw, y1), bg_color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            except Exception as exc:
                logger.error(f"Error processing frame: {exc}", exc_info=True)

            current_occupancy[filename] = projected_points

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                logger.warning("JPEG encoding failed for %s", filename)
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        current_occupancy[filename] = []

# НОВИЙ ЕНДПОІНТ: Збереження зони з React
@app.post("/set_zone/{filename}")
async def set_zone(filename: str, request: Request):
    points = await request.json()
    ensure_data_dirs()
    zones_path = ZONES_JSON_PATH

    data = _load_json_file(zones_path)
    data[_zone_key_for_filename(filename)] = points

    _write_json_file(zones_path, data)

    return {"status": "success"}


@app.post("/upload_floor_plan")
async def upload_floor_plan(file: UploadFile = File(...)):
    ensure_data_dirs()
    with FLOOR_PLAN_PATH.open("wb") as buffer:
        buffer.write(await file.read())
    await file.close()
    return {"status": "uploaded"}


@app.get("/floor_plan")
async def get_floor_plan():
    if not FLOOR_PLAN_PATH.exists():
        raise HTTPException(status_code=404, detail="Floor plan not found")
    return FileResponse(FLOOR_PLAN_PATH)


@app.post("/calibrate_camera/{filename}")
async def calibrate_camera(filename: str, payload: CalibrationPayload):
    ensure_data_dirs()
    data = _load_json_file(CALIBRATION_JSON_PATH)
    data[_calibration_key_for_filename(filename)] = payload.model_dump()
    _write_json_file(CALIBRATION_JSON_PATH, data)
    return {"status": "success"}


@app.get("/calibration/{filename}")
async def get_calibration(filename: str):
    data = _load_json_file(CALIBRATION_JSON_PATH)
    calibration = data.get(_calibration_key_for_filename(filename))
    if calibration is None:
        return {"src_pts": [], "dst_pts": []}
    return calibration

@app.post("/upload_video")
async def upload_video(
    request: Request,
    video: UploadFile = File(...),
    camera_id: int | None = Form(None),
):
    unique_filename = _build_unique_filename(video.filename)
    file_path = UPLOAD_DIR / unique_filename
    with file_path.open("wb") as buffer:
        buffer.write(await video.read())
    await video.close()

    thumbnail_name = _save_video_thumbnail(file_path)
    
    # Додаємо camera_id до URL стріму, щоб бекенд знав, чию зону вантажити
    base_stream_url = str(request.url_for("video_feed", filename=unique_filename))
    stream_url = f"{base_stream_url}?camera_id={camera_id}" if camera_id else base_stream_url
    
    thumbnail_url = (
        str(request.url_for("thumbnails", path=thumbnail_name))
        if thumbnail_name is not None
        else None
    )

    return {
        "status": "uploaded",
        "filename": unique_filename,
        "stream_url": stream_url,
        "thumbnail_url": thumbnail_url,
    }


@app.get("/api/stats")
async def get_stats():
    age_order = ["0-17", "18-25", "26-35", "36-45", "46-55", "56+", "Unknown"]

    try:
        rows = list(customers_collection.find({}, {"_id": 0, "age": 1, "gender": 1}))
    except Exception as exc:
        return {
            "gender_distribution": [],
            "age_groups": [{"label": label, "count": 0} for label in age_order],
            "total_customers": 0,
            "error": f"Could not load analytics: {exc}",
        }

    if not rows:
        return {
            "gender_distribution": [],
            "age_groups": [{"label": label, "count": 0} for label in age_order],
            "total_customers": 0,
        }

    gender_counts: dict[str, int] = {}
    age_counts = {label: 0 for label in age_order}

    for row in rows:
        gender_label = _normalize_gender_label(row.get("gender"))
        gender_counts[gender_label] = gender_counts.get(gender_label, 0) + 1

        age_label = _age_bucket_from_value(row.get("age"))
        age_counts[age_label] = age_counts.get(age_label, 0) + 1

    return {
        "gender_distribution": [
            {"label": label, "count": count}
            for label, count in sorted(gender_counts.items(), key=lambda item: item[0])
        ],
        "age_groups": [{"label": label, "count": age_counts[label]} for label in age_order],
        "total_customers": len(rows),
    }


@app.get("/api/occupancy")
async def get_occupancy():
    safe_occupancy = {
        camera_filename: points if isinstance(points, list) else []
        for camera_filename, points in current_occupancy.items()
    }
    merged_points = _merge_projected_points(safe_occupancy)
    return {"total_customers": len(merged_points), "points": merged_points}

@app.get("/video_feed/{filename}", name="video_feed")
async def video_feed(filename: str, camera_id: int = None):
    video_path = UPLOAD_DIR / filename
    return StreamingResponse(
        generate_yolo_frames(str(video_path), filename, camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)