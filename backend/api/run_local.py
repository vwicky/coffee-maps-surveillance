import cv2
import csv
import json
import logging
import numpy as np
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pymongo import MongoClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from classification.src.face_classifier import FaceClassifier
from app_paths import (
    CALIBRATION_JSON_PATH,
    DATA_ROOT,
    FLOOR_PLAN_PATH,
    LOGS_DIR,
    SNAPSHOTS_DIR,
    THUMBNAILS_DIR,
    UPLOADS_DIR,
    ZONES_JSON_PATH,
    ensure_data_dirs,
)
from reporting.people_report_builder import build_people_final_report, build_visitors_json_report

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

mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
mongo_db = mongo_client["coffee_shop_analytics"]
sessions_collection = mongo_db["sessions"]
current_occupancy: dict[str, list[dict[str, int]]] = {}
LIVE_SESSION_SNAPSHOTS_DIR = SNAPSHOTS_DIR / "live_current"
LIVE_SESSION_ARCHIVE_DIR = SNAPSHOTS_DIR / "processed_sessions"
live_session_lock = threading.Lock()
active_streams = 0
session_finalization_requested = False
live_session_track_times: dict[int, dict[str, object]] = {}


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


def _ensure_live_session_dirs() -> None:
    ensure_data_dirs()
    LIVE_SESSION_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_SESSION_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _compose_live_track_id(camera_id: int | None, track_id: int) -> int:
    camera_component = int(camera_id) if camera_id is not None else 0
    return camera_component * 1_000_000 + int(track_id)


def _write_live_people_times_csv(output_path: Path, track_times: dict[int, dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID",
            "Enter",
            "Exit",
            "Duration_sec",
            "Frames_seen",
            "Max_conf",
            "First_bbox",
            "Last_bbox",
            "Max_area",
            "Snapshots_saved",
            "Is_Staff",
        ])
        for track_id, row in sorted(track_times.items(), key=lambda item: item[0]):
            enter_time = row.get("enter")
            exit_time = row.get("exit")
            if not isinstance(enter_time, datetime) or not isinstance(exit_time, datetime):
                continue

            duration = max(0.0, (exit_time - enter_time).total_seconds())
            writer.writerow([
                track_id,
                enter_time.strftime("%Y-%m-%d %H:%M:%S"),
                exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{duration:.1f}",
                row.get("frames_seen", 0),
                f"{float(row['max_conf']):.3f}" if row.get("max_conf") is not None else "",
                str(row.get("first_bbox", "")),
                str(row.get("last_bbox", "")),
                row.get("max_area", ""),
                row.get("snapshots_saved", 0),
                bool(row.get("is_staff", False)),
            ])


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
) -> Path | None:
    _ensure_live_session_dirs()
    live_track_id = _compose_live_track_id(camera_id, track_id)
    person_dir = LIVE_SESSION_SNAPSHOTS_DIR / str(live_track_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = person_dir / f"shot_{int(time.time() * 1000)}.jpg"
    if not cv2.imwrite(os.fspath(snapshot_path), crop):
        return None
    return snapshot_path


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


def _build_stats_group(rows: list[dict], age_order: list[str]) -> dict:
    gender_counts: dict[str, int] = {}
    age_counts = {label: 0 for label in age_order}
    total_duration_sec = 0.0
    hour_counts: dict[str, int] = {}

    for row in rows:
        gender_label = _normalize_gender_label(row.get("gender"))
        gender_counts[gender_label] = gender_counts.get(gender_label, 0) + 1

        age_label = _age_bucket_from_value(row.get("age"))
        age_counts[age_label] = age_counts.get(age_label, 0) + 1

        try:
            total_duration_sec += float(row.get("duration_sec", 0) or 0)
        except (TypeError, ValueError):
            pass

        hour_value = str(row.get("hour", "") or "").strip()
        if hour_value:
            hour_counts[hour_value] = hour_counts.get(hour_value, 0) + 1

    total_customers = len(rows)
    peak_hour_key = max(hour_counts.items(), key=lambda item: item[1])[0] if hour_counts else "Unknown"

    return {
        "total_customers": total_customers,
        "gender_distribution": [
            {"label": label, "count": count}
            for label, count in sorted(gender_counts.items(), key=lambda item: item[0])
        ],
        "age_groups": [{"label": label, "count": age_counts[label]} for label in age_order],
        "average_duration_sec": (total_duration_sec / total_customers) if total_customers > 0 else 0,
        "peak_hour": f"{peak_hour_key}:00" if peak_hour_key.isdigit() and len(peak_hour_key) == 2 else "Unknown",
    }


def _build_buffered_zone(
    text: str,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    image_width: int,
    image_height: int,
    index: int,
    padding: int = 40,
) -> dict[str, object]:
    safe_name = str(text or "").strip() or f"Detected {index}"
    safe_id = "".join(ch.lower() if ch.isalnum() else "_" for ch in safe_name).strip("_")
    safe_id = safe_id or f"auto_{index}"
    return {
        "id": f"{safe_id}_{index}",
        "name": safe_name,
        "xMin": max(0, int(x_min) - padding),
        "xMax": min(int(image_width), int(x_max) + padding),
        "yMin": max(0, int(y_min) - padding),
        "yMax": min(int(image_height), int(y_max) + padding),
    }


def _extract_floor_plan_contours(image: np.ndarray) -> list[np.ndarray]:
    image_height, image_width = image.shape[:2]
    total_area = max(1, image_width * image_height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        9,
    )
    combined = cv2.bitwise_or(edges, threshold)
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.dilate(combined, kernel, iterations=1)
    contours, _ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1500:
            continue
        if area > 0.8 * total_area:
            continue
        valid_contours.append(contour)
    return valid_contours


def _rotate_point_from_clockwise_image(
    x: float,
    y: float,
    original_height: int,
) -> tuple[int, int]:
    return int(round(y)), max(0, int(round(original_height - x - 1)))


def _rotate_point_from_counterclockwise_image(
    x: float,
    y: float,
    original_width: int,
) -> tuple[int, int]:
    return max(0, int(round(original_width - y - 1))), int(round(x))


def _build_text_box_record(
    index: int,
    text: str,
    points: list[tuple[int, int]],
    confidence: float | None = None,
) -> dict[str, object] | None:
    cleaned_text = str(text or "").strip()
    if not cleaned_text:
        return None
    if len(cleaned_text) < 3 or cleaned_text.isnumeric():
        return None
    if confidence is not None and confidence < 0.5:
        return None

    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return {
        "index": index,
        "text": cleaned_text,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "cx": int((x_min + x_max) / 2),
        "cy": int((y_min + y_max) / 2),
        "confidence": confidence,
    }


def _dedupe_text_boxes(text_boxes: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []

    for text_box in text_boxes:
        is_duplicate = False
        for existing_index, existing in enumerate(deduped):
            same_text = str(text_box["text"]).lower() == str(existing["text"]).lower()
            close_x = abs(int(text_box["cx"]) - int(existing["cx"])) <= 30
            close_y = abs(int(text_box["cy"]) - int(existing["cy"])) <= 30
            if same_text and close_x and close_y:
                existing_conf = float(existing.get("confidence") or 0.0)
                current_conf = float(text_box.get("confidence") or 0.0)
                if current_conf > existing_conf:
                    deduped[existing_index] = text_box
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(text_box)

    for index, text_box in enumerate(deduped, start=1):
        text_box["index"] = index

    return deduped


def _extract_easyocr_text_boxes(
    image: np.ndarray,
    *,
    rotation_mode: str = "original",
    original_width: int | None = None,
    original_height: int | None = None,
) -> list[dict[str, object]]:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)
    detections = reader.readtext(image)
    text_boxes: list[dict[str, object]] = []

    for index, detection in enumerate(detections, start=1):
        if len(detection) < 2:
            continue
        bbox, text = detection[0], str(detection[1])
        confidence = float(detection[2]) if len(detection) > 2 and detection[2] is not None else None
        points = [(int(point[0]), int(point[1])) for point in bbox]
        if rotation_mode == "cw":
            if original_height is None:
                raise ValueError("original_height is required for clockwise OCR coordinates")
            points = [_rotate_point_from_clockwise_image(x, y, original_height) for x, y in points]
        elif rotation_mode == "ccw":
            if original_width is None:
                raise ValueError("original_width is required for counterclockwise OCR coordinates")
            points = [_rotate_point_from_counterclockwise_image(x, y, original_width) for x, y in points]

        text_box = _build_text_box_record(index, text, points, confidence=confidence)
        if text_box is not None:
            text_boxes.append(text_box)

    return text_boxes


def _extract_pytesseract_text_boxes(
    image: np.ndarray,
    *,
    rotation_mode: str = "original",
    original_width: int | None = None,
    original_height: int | None = None,
) -> list[dict[str, object]]:
    import pytesseract

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
    grouped: dict[tuple[int, int, int], dict[str, object]] = {}
    total_items = len(data.get("text", []))

    for idx in range(total_items):
        text = str(data["text"][idx] or "").strip()
        if not text:
            continue

        try:
            confidence = float(data["conf"][idx])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < 0:
            continue

        key = (
            int(data["block_num"][idx]),
            int(data["par_num"][idx]),
            int(data["line_num"][idx]),
        )
        left = int(data["left"][idx])
        top = int(data["top"][idx])
        width = int(data["width"][idx])
        height = int(data["height"][idx])

        group = grouped.setdefault(
            key,
            {
                "texts": [],
                "confidences": [],
                "x_min": left,
                "x_max": left + width,
                "y_min": top,
                "y_max": top + height,
            },
        )
        group["texts"].append(text)
        group["confidences"].append(confidence)
        group["x_min"] = min(int(group["x_min"]), left)
        group["x_max"] = max(int(group["x_max"]), left + width)
        group["y_min"] = min(int(group["y_min"]), top)
        group["y_max"] = max(int(group["y_max"]), top + height)

    text_boxes: list[dict[str, object]] = []
    for index, group in enumerate(grouped.values(), start=1):
        label = " ".join(group["texts"])
        x_min = int(group["x_min"])
        x_max = int(group["x_max"])
        y_min = int(group["y_min"])
        y_max = int(group["y_max"])
        points = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]
        if rotation_mode == "cw":
            if original_height is None:
                raise ValueError("original_height is required for clockwise OCR coordinates")
            points = [_rotate_point_from_clockwise_image(x, y, original_height) for x, y in points]
        elif rotation_mode == "ccw":
            if original_width is None:
                raise ValueError("original_width is required for counterclockwise OCR coordinates")
            points = [_rotate_point_from_counterclockwise_image(x, y, original_width) for x, y in points]

        confidence_values = [float(value) for value in group["confidences"] if float(value) >= 0]
        avg_confidence = (sum(confidence_values) / len(confidence_values) / 100.0) if confidence_values else None
        text_box = _build_text_box_record(index, label, points, confidence=avg_confidence)
        if text_box is not None:
            text_boxes.append(text_box)

    return text_boxes


def _build_zone_from_text_and_contours(
    text_box: dict[str, object],
    contours: list[np.ndarray],
    image_width: int,
    image_height: int,
) -> dict[str, object]:
    text = str(text_box["text"])
    index = int(text_box["index"])
    center = (float(text_box["cx"]), float(text_box["cy"]))

    selected_contour: np.ndarray | None = None
    selected_area = float("inf")

    for contour in contours:
        if cv2.pointPolygonTest(contour, center, False) < 0:
            continue

        contour_area = cv2.contourArea(contour)
        if contour_area < selected_area:
            selected_contour = contour
            selected_area = contour_area

    if selected_contour is not None:
        x, y, w, h = cv2.boundingRect(selected_contour)
        return _build_buffered_zone(
            text,
            x,
            x + w,
            y,
            y + h,
            image_width,
            image_height,
            index,
            padding=30,
        )

    return _build_buffered_zone(
        text,
        int(text_box["x_min"]),
        int(text_box["x_max"]),
        int(text_box["y_min"]),
        int(text_box["y_max"]),
        image_width,
        image_height,
        index,
        padding=80,
    )


def _detect_floor_plan_zones(image: np.ndarray) -> list[dict[str, object]]:
    errors: list[str] = []
    text_boxes: list[dict[str, object]] = []
    image_height, image_width = image.shape[:2]
    contours = _extract_floor_plan_contours(image)
    rotated_cw_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_ccw_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    try:
        text_boxes = _dedupe_text_boxes(
            _extract_easyocr_text_boxes(image)
            + _extract_easyocr_text_boxes(
                rotated_cw_image,
                rotation_mode="cw",
                original_height=image_height,
                original_width=image_width,
            )
            + _extract_easyocr_text_boxes(
                rotated_ccw_image,
                rotation_mode="ccw",
                original_height=image_height,
                original_width=image_width,
            )
        )
    except ImportError:
        errors.append("easyocr is not installed")
    except Exception as exc:
        logger.warning("easyocr floor-plan analysis failed: %s", exc)
        errors.append("easyocr failed")

    if not text_boxes:
        try:
            text_boxes = _dedupe_text_boxes(
                _extract_pytesseract_text_boxes(image)
                + _extract_pytesseract_text_boxes(
                    rotated_cw_image,
                    rotation_mode="cw",
                    original_height=image_height,
                    original_width=image_width,
                )
                + _extract_pytesseract_text_boxes(
                    rotated_ccw_image,
                    rotation_mode="ccw",
                    original_height=image_height,
                    original_width=image_width,
                )
            )
        except ImportError:
            errors.append("pytesseract is not installed")
        except Exception as exc:
            logger.warning("pytesseract floor-plan analysis failed: %s", exc)
            errors.append("pytesseract failed")

    if text_boxes:
        return [
            _build_zone_from_text_and_contours(text_box, contours, image_width, image_height)
            for text_box in text_boxes
        ]

    if errors:
        logger.warning("Floor-plan OCR unavailable or produced no zones: %s", "; ".join(errors))
    return []


def _session_date_key(session_id: object) -> str:
    session_value = str(session_id or "").strip()
    parts = session_value.split("_")
    if len(parts) >= 3:
        try:
            return datetime.strptime(parts[1], "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    return "Unknown Date"


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
    occupancy_by_camera: dict[str, list[dict[str, object]]],
    distance_threshold: float = 50.0,
) -> list[dict[str, object]]:
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
                        "is_staff": point.get("is_staff", False),
                        "cameras": {camera_filename},
                    }
                )
                continue

            count = int(best_cluster["count"])
            best_cluster["x"] = (float(best_cluster["x"]) * count + x) / (count + 1)
            best_cluster["y"] = (float(best_cluster["y"]) * count + y) / (count + 1)
            best_cluster["count"] = count + 1
            best_cluster["is_staff"] = bool(best_cluster.get("is_staff", False) or point.get("is_staff", False))
            best_cluster["cameras"].add(camera_filename)

    return [
        {
            "x": int(round(float(cluster["x"]))),
            "y": int(round(float(cluster["y"]))),
            "is_staff": bool(cluster.get("is_staff", False)),
        }
        for cluster in clusters
    ]


def _finalize_live_session(session_id: str) -> None:
    global live_session_track_times, session_finalization_requested
    archive_dir: Path | None = None

    try:
        deadline = time.time() + 10.0
        while True:
            with live_session_lock:
                streams_left = active_streams
            if streams_left == 0 or time.time() >= deadline:
                break
            time.sleep(0.25)

        with live_session_lock:
            _ensure_live_session_dirs()
            has_snapshots = any(path.is_dir() for path in LIVE_SESSION_SNAPSHOTS_DIR.iterdir())
            if not has_snapshots:
                logger.info("No live snapshots found to finalize for %s", session_id)
                return

            people_times_csv = LIVE_SESSION_SNAPSHOTS_DIR / "people_times.csv"
            _write_live_people_times_csv(people_times_csv, live_session_track_times)
            archive_dir = LIVE_SESSION_ARCHIVE_DIR / session_id
            shutil.move(os.fspath(LIVE_SESSION_SNAPSHOTS_DIR), os.fspath(archive_dir))
            LIVE_SESSION_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

        analyzer = FaceClassifier(ctx_id=-1, load_previous_faces=False)
        analyzer.mongo_metadata_manager.create_session(
            session_id,
            str(archive_dir),
            "live session finalization",
        )

        people_times_csv = archive_dir / "people_times.csv"
        analyzer.analyze_folder(
            str(archive_dir),
            session_id,
            people_times_csv_path=str(people_times_csv),
        )

        report_path = build_people_final_report(
            mongo_manager=analyzer.mongo_metadata_manager,
            session_id=session_id,
            people_times_csv_path=str(people_times_csv),
            output_csv_path=str(LOGS_DIR / f"people_final_report_{session_id}.csv"),
        )
        build_visitors_json_report(
            merged_csv_path=report_path,
            output_json_path=str(LOGS_DIR / f"visitors_report_{session_id}.json"),
        )
        logger.info("Finished live session finalization for %s", session_id)
    except Exception:
        logger.exception("Failed to finalize live session %s", session_id)
    finally:
        if archive_dir is not None:
            shutil.rmtree(archive_dir, ignore_errors=True)
        with live_session_lock:
            live_session_track_times = {}
            session_finalization_requested = False


def generate_yolo_frames(video_path, filename, camera_id=None):
    from ultralytics import YOLO

    global active_streams, live_session_track_times
    logger.info("Starting stream for %s", filename)
    print(f"▶️ STREAM STARTED: {filename}")
    _ensure_live_session_dirs()
    with live_session_lock:
        if active_streams == 0:
            live_session_track_times.clear()
        active_streams += 1
    model = YOLO("yolov8s.pt")
    processed_customer_ids = set()
    track_zone_stats = {}
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
            projected_points: list[dict[str, object]] = []
            try:
                current_time = datetime.now()
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
                        if tid is not None:
                            tid = int(tid)
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        bottom_y = int(y2)

                        in_zone_now = False
                        if barista_zone is not None:
                            in_zone_now = (
                                cv2.pointPolygonTest(barista_zone, (center_x, center_y), False) >= 0
                                or cv2.pointPolygonTest(barista_zone, (center_x, bottom_y), False) >= 0
                            )

                        if tid is not None:
                            if tid not in track_zone_stats:
                                track_zone_stats[tid] = {"total_frames": 0, "in_zone_frames": 0}

                            track_zone_stats[tid]["total_frames"] += 1
                            if in_zone_now:
                                track_zone_stats[tid]["in_zone_frames"] += 1

                            ratio = (
                                track_zone_stats[tid]["in_zone_frames"]
                                / track_zone_stats[tid]["total_frames"]
                            )
                            is_staff = ratio >= 0.5
                        else:
                            is_staff = in_zone_now

                        live_track_id = _compose_live_track_id(camera_id, tid) if tid is not None else None
                        confidence = None
                        if hasattr(r.boxes, "conf") and r.boxes.conf is not None and i < len(r.boxes.conf):
                            confidence = float(r.boxes.conf[i])

                        if live_track_id is not None:
                            with live_session_lock:
                                track_time = live_session_track_times.setdefault(
                                    live_track_id,
                                    {
                                        "enter": current_time,
                                        "exit": current_time,
                                        "frames_seen": 0,
                                        "max_conf": None,
                                        "first_bbox": [x1, y1, x2, y2],
                                        "last_bbox": [x1, y1, x2, y2],
                                        "max_area": 0,
                                        "snapshots_saved": 0,
                                        "is_staff": False,
                                    },
                                )
                                track_time["exit"] = current_time
                                track_time["frames_seen"] = int(track_time.get("frames_seen", 0)) + 1
                                track_time["last_bbox"] = [x1, y1, x2, y2]
                                area = max(0, (x2 - x1) * (y2 - y1))
                                track_time["max_area"] = max(int(track_time.get("max_area", 0) or 0), area)
                                if confidence is not None:
                                    prev_conf = track_time.get("max_conf")
                                    track_time["max_conf"] = (
                                        confidence
                                        if prev_conf is None
                                        else max(float(prev_conf), confidence)
                                    )
                                track_time["is_staff"] = bool(is_staff)

                        if is_staff:
                            label = "Staff"
                            bg_color = (0, 165, 255)
                        else:
                            label = "Customer"
                            bg_color = (0, 255, 0)

                        map_x = None
                        map_y = None
                        if homography_matrix is not None:
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
                            projected_points.append({"x": map_x, "y": map_y, "is_staff": bool(is_staff)})

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bg_color, 2)

                        if not is_staff and tid is not None:
                            with live_session_lock:
                                should_save_snapshot = not session_finalization_requested

                            if should_save_snapshot and tid not in processed_customer_ids:
                                crop = _safe_crop_person(r.orig_img, (x1, y1, x2, y2))
                                if crop is not None:
                                    snapshot_path = _save_customer_snapshot(crop, camera_id, tid)
                                    if snapshot_path is not None:
                                        processed_customer_ids.add(tid)
                                        if live_track_id is not None:
                                            with live_session_lock:
                                                track_time = live_session_track_times.setdefault(
                                                    live_track_id,
                                                    {
                                                        "enter": current_time,
                                                        "exit": current_time,
                                                        "frames_seen": 0,
                                                        "max_conf": None,
                                                        "first_bbox": [x1, y1, x2, y2],
                                                        "last_bbox": [x1, y1, x2, y2],
                                                        "max_area": max(0, (x2 - x1) * (y2 - y1)),
                                                        "snapshots_saved": 0,
                                                        "is_staff": False,
                                                    },
                                                )
                                                track_time["snapshots_saved"] = 1

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
        with live_session_lock:
            _ensure_live_session_dirs()
            _write_live_people_times_csv(LIVE_SESSION_SNAPSHOTS_DIR / "people_times.csv", live_session_track_times)
            active_streams = max(0, active_streams - 1)

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


@app.post("/api/analyze_floor_plan")
async def analyze_floor_plan(file: UploadFile | None = File(None)):
    if file is not None:
        payload = await file.read()
        await file.close()
    else:
        if not FLOOR_PLAN_PATH.exists():
            raise HTTPException(status_code=404, detail="Floor plan not found")
        payload = FLOOR_PLAN_PATH.read_bytes()

    if not payload:
        raise HTTPException(status_code=400, detail="Floor plan image is empty")

    image_array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode floor plan image")

    zones = _detect_floor_plan_zones(image)
    return {"zones": zones}


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


@app.post("/api/end_session")
async def end_session(background_tasks: BackgroundTasks):
    global session_finalization_requested

    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with live_session_lock:
        _ensure_live_session_dirs()
        has_snapshots = any(path.is_dir() for path in LIVE_SESSION_SNAPSHOTS_DIR.iterdir())
        if session_finalization_requested:
            return {"status": "already_processing"}
        if not has_snapshots:
            return {"status": "no_snapshots"}
        session_finalization_requested = True

    background_tasks.add_task(_finalize_live_session, session_id)
    return {
        "status": "success",
        "message": "Session finalization started in background",
        "session_id": session_id,
    }


@app.get("/api/stats")
async def get_stats():
    age_order = ["0-17", "18-25", "26-35", "36-45", "46-55", "56+", "Unknown"]
    empty_group = {
        "total_customers": 0,
        "gender_distribution": [],
        "age_groups": [{"label": label, "count": 0} for label in age_order],
        "average_duration_sec": 0,
        "peak_hour": "Unknown",
    }

    try:
        sessions = list(sessions_collection.find({}, {"_id": 0, "session_id": 1, "faces": 1}))
    except Exception as exc:
        logger.warning("Could not load analytics stats: %s", exc)
        return {"overall": empty_group, "by_date": {}}

    if not sessions:
        return {"overall": empty_group, "by_date": {}}

    overall_rows: list[dict] = []
    rows_by_date: dict[str, list[dict]] = {}

    for session in sessions:
        faces = session.get("faces", [])
        if not isinstance(faces, list):
            continue

        for face in faces:
            if not isinstance(face, dict):
                continue

            additional_metadata = face.get("additional_metadata", {}) or {}
            if bool(additional_metadata.get("is_staff", False)):
                continue

            enter_time = str(additional_metadata.get("enter_time", "") or "").strip()
            date_key = enter_time[:10] if len(enter_time) >= 10 else "Unknown Date"
            hour_str = enter_time[11:13] if len(enter_time) >= 13 else "Unknown"

            face_row = {
                "age": face.get("classification_result", {}).get("age"),
                "gender": face.get("classification_result", {}).get("sex"),
                "duration_sec": additional_metadata.get("duration_sec", 0),
                "hour": hour_str,
            }
            overall_rows.append(face_row)
            rows_by_date.setdefault(date_key, []).append(face_row)

    if not overall_rows:
        return {"overall": empty_group, "by_date": {}}

    return {
        "overall": _build_stats_group(overall_rows, age_order),
        "by_date": {
            date_key: _build_stats_group(date_rows, age_order)
            for date_key, date_rows in sorted(rows_by_date.items(), key=lambda item: item[0], reverse=True)
        },
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