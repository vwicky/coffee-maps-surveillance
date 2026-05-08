import cv2
import csv
import json
import jwt
import logging
import numpy as np
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

from bson import ObjectId
from pymongo import MongoClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
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
CAFE_UPLOADS_DIR = BACKEND_ROOT / "uploads"
os.makedirs(CAFE_UPLOADS_DIR, exist_ok=True)

# Налаштування CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_data_dirs()
UPLOAD_DIR = UPLOADS_DIR
app.mount("/uploads", StaticFiles(directory=os.fspath(CAFE_UPLOADS_DIR)), name="uploads")
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")

mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
mongo_db = mongo_client["coffee_shop_analytics"]
sessions_collection = mongo_db["sessions"]
users_collection = mongo_db["users"]
cafes_collection = mongo_db["cafes"]
cafe_layouts_collection = mongo_db["cafe_layouts"]
zones_collection = mongo_db["zones"]
camera_configs_collection = mongo_db["camera_configs"]
current_occupancy: dict[str, dict[str, list[dict[str, int]]]] = {}
LIVE_SESSION_SNAPSHOTS_DIR = SNAPSHOTS_DIR / "live_current"
LIVE_SESSION_ARCHIVE_DIR = SNAPSHOTS_DIR / "processed_sessions"
live_session_lock = threading.Lock()
active_streams: dict[str, int] = {}
# Filenames currently generating MJPEG in ``generate_yolo_frames`` (per cafe).
streaming_filenames: defaultdict[str, set[str]] = defaultdict(set)
session_finalization_requested: dict[str, bool] = {}
live_session_track_times: dict[str, dict[int, dict[str, object]]] = {}
POSITION_SAMPLE_INTERVAL_SEC = 1.0
# Expand zone rectangles when serving zones to clients / classification (seating around furniture).
ZONE_PADDING = 25
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production-coffee-maps-auth-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class Point(BaseModel):
    x: float
    y: float


class CalibrationPayload(BaseModel):
    src_pts: list[Point]
    dst_pts: list[Point]


class FloorPlanZone(BaseModel):
    id: str
    name: str
    xMin: float
    xMax: float
    yMin: float
    yMax: float


class UserRegister(BaseModel):
    full_name: str
    email: str
    password: str
    cafe_name: str
    cafe_address: str


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


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


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = users_collection.find_one({"email": _normalize_email(email)})
    if user is None:
        raise credentials_exception

    return user


def _get_user_from_token(token: str):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = users_collection.find_one({"email": _normalize_email(email)})
    if user is None:
        raise credentials_exception

    return user


def _get_cafe_for_user(current_user):
    user_cafe = cafes_collection.find_one({"owner_id": str(current_user["_id"])})
    if not user_cafe:
        raise HTTPException(status_code=404, detail="Cafe not found")
    return user_cafe


def _get_cafe_by_id(cafe_id: str):
    try:
        return cafes_collection.find_one({"_id": ObjectId(cafe_id)})
    except Exception:
        return cafes_collection.find_one({"_id": cafe_id})


def _get_cafe_live_snapshots_dir(cafe_id: str) -> Path:
    return LIVE_SESSION_SNAPSHOTS_DIR / str(cafe_id)


def _get_cafe_live_archive_dir(cafe_id: str) -> Path:
    return LIVE_SESSION_ARCHIVE_DIR / str(cafe_id)


def _get_cafe_floor_plan_path(cafe: dict | None) -> Path | None:
    floor_plan_url = str((cafe or {}).get("floor_plan_url", "") or "").strip()
    if not floor_plan_url:
        return None

    file_name = Path(floor_plan_url).name
    if not file_name:
        return None

    file_path = CAFE_UPLOADS_DIR / file_name
    if not file_path.exists():
        return None

    return file_path


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


def _ensure_live_session_dirs(cafe_id: str) -> None:
    ensure_data_dirs()
    _get_cafe_live_snapshots_dir(cafe_id).mkdir(parents=True, exist_ok=True)
    _get_cafe_live_archive_dir(cafe_id).mkdir(parents=True, exist_ok=True)


def _compose_live_track_id(camera_id: int | None, track_id: int) -> int:
    camera_component = int(camera_id) if camera_id is not None else 0
    return camera_component * 1_000_000 + int(track_id)


def _serialize_position_samples(samples: object) -> str:
    if not isinstance(samples, list):
        return "[]"
    return json.dumps(samples, ensure_ascii=False, separators=(",", ":"))


def _normalize_zone_record(raw_zone: object, index: int) -> dict[str, object] | None:
    if not isinstance(raw_zone, dict):
        return None

    try:
        x_min = int(float(raw_zone.get("xMin")))
        x_max = int(float(raw_zone.get("xMax")))
        y_min = int(float(raw_zone.get("yMin")))
        y_max = int(float(raw_zone.get("yMax")))
    except (TypeError, ValueError):
        return None

    zone_name = str(raw_zone.get("name", "") or "").strip() or f"Zone {index}"
    zone_id = str(raw_zone.get("id", "") or "").strip() or f"zone_{index}"
    return {
        "id": zone_id,
        "name": zone_name,
        "xMin": min(x_min, x_max),
        "xMax": max(x_min, x_max),
        "yMin": min(y_min, y_max),
        "yMax": max(y_min, y_max),
    }


def _save_floor_plan_map_zones(cafe_id: str, zones: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized = [
        zone
        for index, zone in enumerate(
            (_normalize_zone_record(raw_zone, index) for index, raw_zone in enumerate(zones, start=1)),
            start=1,
        )
        if zone is not None
    ]
    zones_collection.update_one(
        {"cafe_id": str(cafe_id)},
        {
            "$set": {
                "zones": normalized,
                "updated_at": datetime.utcnow(),
            },
            "$setOnInsert": {
                "cafe_id": str(cafe_id),
                "created_at": datetime.utcnow(),
            },
        },
        upsert=True,
    )
    return normalized


def _load_floor_plan_map_zones(cafe_id: str) -> list[dict[str, object]]:
    document = zones_collection.find_one({"cafe_id": str(cafe_id)}, {"_id": 0, "zones": 1}) or {}
    stored_zones = document.get("zones")
    if not isinstance(stored_zones, list):
        legacy_document = (
            cafe_layouts_collection.find_one({"cafe_id": str(cafe_id)}, {"_id": 0, "map_zones": 1}) or {}
        )
        stored_zones = legacy_document.get("map_zones")
    if not isinstance(stored_zones, list):
        return []

    return [
        zone
        for index, zone in enumerate(
            (_normalize_zone_record(raw_zone, index) for index, raw_zone in enumerate(stored_zones, start=1)),
            start=1,
        )
        if zone is not None
    ]


def _expand_zone_bounds_for_seating(zones: list[dict[str, object]], pad: int = ZONE_PADDING) -> list[dict[str, object]]:
    """Return a copy of zones with x/y bounds expanded by ``pad`` px (mins floored at 0)."""
    if pad <= 0:
        return [dict(z) for z in zones]
    expanded: list[dict[str, object]] = []
    for z in zones:
        try:
            xm = int(float(z["xMin"]))
            xM = int(float(z["xMax"]))
            ym = int(float(z["yMin"]))
            yM = int(float(z["yMax"]))
        except (KeyError, TypeError, ValueError):
            expanded.append(dict(z))
            continue
        nz = dict(z)
        nz["xMin"] = max(0, xm - pad)
        nz["yMin"] = max(0, ym - pad)
        nz["xMax"] = xM + pad
        nz["yMax"] = yM + pad
        expanded.append(nz)
    return expanded


def _resolve_zone_name(x: int | float | None, y: int | float | None, zones: list[dict[str, object]]) -> str:
    if x is None or y is None:
        return ""

    px = float(x)
    py = float(y)

    for zone in zones:
        try:
            if float(zone["xMin"]) <= px <= float(zone["xMax"]) and float(zone["yMin"]) <= py <= float(zone["yMax"]):
                return str(zone.get("name", "") or "")
        except (KeyError, TypeError, ValueError):
            continue
    return ""


def _append_position_sample(
    track_time: dict[str, object],
    *,
    sample_time: datetime,
    x: int,
    y: int,
    zone_name: str,
    interval_sec: float = POSITION_SAMPLE_INTERVAL_SEC,
) -> None:
    last_sample_time = track_time.get("_last_position_sample_time")
    if isinstance(last_sample_time, datetime):
        elapsed_sec = (sample_time - last_sample_time).total_seconds()
        if elapsed_sec < interval_sec:
            return

    samples = track_time.setdefault("position_samples", [])
    if not isinstance(samples, list):
        samples = []
        track_time["position_samples"] = samples

    samples.append(
        {
            "timestamp": sample_time.strftime("%Y-%m-%d %H:%M:%S"),
            "x": int(x),
            "y": int(y),
            "zone_name": zone_name,
        }
    )
    track_time["_last_position_sample_time"] = sample_time
    track_time["zone_name"] = zone_name or str(track_time.get("zone_name", "") or "")


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
            "Zone_Name",
            "Position_Samples",
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
                str(row.get("zone_name", "") or ""),
                _serialize_position_samples(row.get("position_samples", [])),
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
    cafe_id: str,
    camera_id: int | None,
    track_id: int,
) -> Path | None:
    _ensure_live_session_dirs(cafe_id)
    live_track_id = _compose_live_track_id(camera_id, track_id)
    person_dir = _get_cafe_live_snapshots_dir(cafe_id) / str(live_track_id)
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


def _format_duration_label(seconds: object) -> str:
    try:
        safe_seconds = max(0, int(round(float(seconds or 0))))
    except (TypeError, ValueError):
        safe_seconds = 0

    minutes = safe_seconds // 60
    remaining_seconds = safe_seconds % 60
    return f"{minutes} хв {remaining_seconds} с"


def _parse_sample_timestamp(raw_value: object) -> datetime | None:
    value = str(raw_value or "").strip()
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def _matches_stats_filters(
    dt_value: datetime | None,
    *,
    filter_date,
    range_start_date,
    start_time_value,
    end_time_value,
) -> bool:
    if dt_value is None:
        return False

    if filter_date is not None and dt_value.date() != filter_date:
        return False
    if range_start_date is not None and dt_value.date() < range_start_date:
        return False
    if start_time_value is not None and end_time_value is not None:
        record_time = dt_value.time()
        if record_time < start_time_value or record_time > end_time_value:
            return False
    return True


def _face_duration_sec(face: dict) -> float:
    meta = face.get("additional_metadata", {}) or {}
    raw = face.get("total_duration_sec")
    if raw is None:
        raw = meta.get("duration_sec")
    try:
        return max(0.0, float(raw or 0))
    except (TypeError, ValueError):
        return 0.0


def _parse_query_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_query_time_range_param(time_range: str | None):
    if not time_range or "-" not in time_range:
        return None, None
    try:
        raw_start, raw_end = [part.strip() for part in time_range.split("-", 1)]
        return (
            datetime.strptime(raw_start, "%H:%M").time(),
            datetime.strptime(raw_end, "%H:%M").time(),
        )
    except ValueError:
        return None, None


def _percent_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return round(((current - previous) / previous) * 100, 1)


def _aggregate_visitor_faces_for_period(
    sessions: list[dict],
    *,
    period_start: date | None,
    period_end: date | None,
    start_time_value,
    end_time_value,
) -> tuple[int, float, str | None, int | None]:
    hour_counts: dict[int, int] = {}
    durations: list[float] = []

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

            enter_dt = _parse_sample_timestamp(str(additional_metadata.get("enter_time", "") or "").strip())
            if enter_dt is None:
                continue

            d = enter_dt.date()
            if period_start is not None and period_end is not None:
                if d < period_start or d > period_end:
                    continue
            elif period_start is not None or period_end is not None:
                continue

            if start_time_value is not None and end_time_value is not None:
                record_time = enter_dt.time()
                if record_time < start_time_value or record_time > end_time_value:
                    continue

            durations.append(_face_duration_sec(face))
            hour_counts[enter_dt.hour] = hour_counts.get(enter_dt.hour, 0) + 1

    n = len(durations)
    avg_sec = (sum(durations) / n) if n > 0 else 0.0
    peak_hour = None
    peak_clients = None
    if hour_counts:
        best_hour, best_count = max(hour_counts.items(), key=lambda item: item[1])
        peak_hour = f"{best_hour:02d}:00"
        peak_clients = int(best_count)

    return n, avg_sec, peak_hour, peak_clients


def _require_analytics_period_or_all(
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[date | None, date | None]:
    d_start = _parse_query_date(start_date)
    d_end = _parse_query_date(end_date)
    if (d_start is None) ^ (d_end is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Надайте обидві дати start_date та end_date або жодної.",
        )
    if d_start is not None and d_end is not None and d_end < d_start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_date не може бути раніше за start_date.",
        )
    return d_start, d_end


def _sessions_for_cafe(cafe_id: str) -> list[dict]:
    """Load session documents for one cafe (multi-tenant isolation)."""
    return list(sessions_collection.find({"cafe_id": cafe_id}, {"_id": 0, "faces": 1}))


def _visitor_face_rows(
    sessions: list[dict],
    *,
    period_start: date | None,
    period_end: date | None,
    start_time_value,
    end_time_value,
) -> list[dict[str, object]]:
    """One row per visitor (face) with visit day/hour and demographics. Uses enter_time on each face."""
    rows: list[dict[str, object]] = []
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

            enter_dt = _parse_sample_timestamp(str(additional_metadata.get("enter_time", "") or "").strip())
            if enter_dt is None:
                continue

            d = enter_dt.date()
            if period_start is not None and period_end is not None:
                if d < period_start or d > period_end:
                    continue
            elif period_start is not None or period_end is not None:
                continue

            if start_time_value is not None and end_time_value is not None:
                record_time = enter_dt.time()
                if record_time < start_time_value or record_time > end_time_value:
                    continue

            clf = face.get("classification_result", {}) or {}
            rows.append(
                {
                    "day": d,
                    "hour": enter_dt.hour,
                    "duration_sec": _face_duration_sec(face),
                    "age": clf.get("age"),
                    "sex": clf.get("sex"),
                }
            )
    return rows


def _analytics_dynamics_series(
    rows: list[dict[str, object]],
    *,
    view: str,
) -> list[dict[str, object]]:
    if view == "days":
        counts: defaultdict[date, int] = defaultdict(int)
        for r in rows:
            counts[r["day"]] += 1  # type: ignore[index]
        return [
            {"date": f"{d.day:02d}-{d.month:02d}", "customers": counts[d]} for d in sorted(counts.keys())
        ]

    counts_h = [0] * 24
    for r in rows:
        h = int(r["hour"])  # type: ignore[arg-type]
        counts_h[h] += 1
    return [{"time": f"{h:02d}:00", "customers": counts_h[h]} for h in range(24)]


def _analytics_demographics_from_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    age_order = ["0-17", "18-25", "26-35", "36-45", "46-55", "56+"]
    buckets = {ag: {"Чоловіки": 0, "Жінки": 0} for ag in age_order}
    for r in rows:
        ag = _age_bucket_from_value(r.get("age"))
        if ag not in buckets:
            continue
        gl = _normalize_gender_label(r.get("sex"))
        if gl == "Male":
            buckets[ag]["Чоловіки"] += 1
        elif gl == "Female":
            buckets[ag]["Жінки"] += 1
    return [{"age_group": ag, **buckets[ag]} for ag in age_order]


def _analytics_daily_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_day: defaultdict[date, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)  # type: ignore[index]

    ordered_days = sorted(by_day.keys(), reverse=True)[:120]
    out: list[dict[str, object]] = []
    for d in ordered_days:
        day_rows = by_day[d]
        total = len(day_rows)
        avg_sec = sum(float(r["duration_sec"]) for r in day_rows) / total if total else 0.0
        hour_counts: defaultdict[int, int] = defaultdict(int)
        for r in day_rows:
            hour_counts[int(r["hour"])] += 1
        peak_str = "—"
        if hour_counts:
            peak_h = max(hour_counts.items(), key=lambda item: item[1])[0]
            peak_str = f"{peak_h:02d}:00"

        m_count = f_count = 0
        age_counts: defaultdict[str, int] = defaultdict(int)
        for r in day_rows:
            gl = _normalize_gender_label(r.get("sex"))
            if gl == "Male":
                m_count += 1
            elif gl == "Female":
                f_count += 1
            age_counts[_age_bucket_from_value(r.get("age"))] += 1

        top_age = "—"
        if age_counts:
            top_age = max(age_counts.items(), key=lambda item: item[1])[0]

        out.append(
            {
                "date": d.isoformat(),
                "total": total,
                "avg_time": _format_duration_label(avg_sec),
                "peak": peak_str,
                "gender": f"Ч: {m_count}, Ж: {f_count}",
                "top_age": top_age,
            }
        )
    return out


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


def _load_zone_for_filename(cafe_id: str, filename: str):
    document = camera_configs_collection.find_one(
        {"cafe_id": str(cafe_id), "filename": filename},
        {"_id": 0, "zone_points": 1},
    ) or {}
    zone_points = document.get("zone_points")
    return zone_points if isinstance(zone_points, list) else None


def _save_zone_for_filename(cafe_id: str, filename: str, points: object) -> list[dict[str, object]]:
    safe_points = points if isinstance(points, list) else []
    camera_configs_collection.update_one(
        {"cafe_id": str(cafe_id), "filename": filename},
        {
            "$set": {
                "zone_points": safe_points,
                "updated_at": datetime.utcnow(),
            },
            "$setOnInsert": {"created_at": datetime.utcnow()},
        },
        upsert=True,
    )
    return safe_points


def _load_calibration_for_filename(cafe_id: str, filename: str):
    document = camera_configs_collection.find_one(
        {"cafe_id": str(cafe_id), "filename": filename},
        {"_id": 0, "calibration": 1},
    ) or {}
    calibration = document.get("calibration")
    return calibration if isinstance(calibration, dict) else None


def _save_calibration_for_filename(cafe_id: str, filename: str, calibration: dict[str, object]) -> None:
    camera_configs_collection.update_one(
        {"cafe_id": str(cafe_id), "filename": filename},
        {
            "$set": {
                "calibration": calibration,
                "updated_at": datetime.utcnow(),
            },
            "$setOnInsert": {"created_at": datetime.utcnow()},
        },
        upsert=True,
    )


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


def _finalize_live_session(session_id: str, cafe_id: str) -> None:
    archive_dir: Path | None = None

    try:
        deadline = time.time() + 10.0
        while True:
            with live_session_lock:
                streams_left = active_streams.get(cafe_id, 0)
            if streams_left == 0 or time.time() >= deadline:
                break
            time.sleep(0.25)

        with live_session_lock:
            _ensure_live_session_dirs(cafe_id)
            snapshots_dir = _get_cafe_live_snapshots_dir(cafe_id)
            archive_root = _get_cafe_live_archive_dir(cafe_id)
            has_snapshots = any(path.is_dir() for path in snapshots_dir.iterdir())
            if not has_snapshots:
                logger.info("No live snapshots found to finalize for %s", session_id)
                return

            people_times_csv = snapshots_dir / "people_times.csv"
            _write_live_people_times_csv(people_times_csv, live_session_track_times.get(cafe_id, {}))
            archive_dir = archive_root / session_id
            shutil.move(os.fspath(snapshots_dir), os.fspath(archive_dir))
            snapshots_dir.mkdir(parents=True, exist_ok=True)

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
        sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"cafe_id": str(cafe_id)}},
        )
        logger.info("Finished live session finalization for %s", session_id)
    except Exception:
        logger.exception("Failed to finalize live session %s", session_id)
    finally:
        if archive_dir is not None:
            shutil.rmtree(archive_dir, ignore_errors=True)
        with live_session_lock:
            live_session_track_times[str(cafe_id)] = {}
            session_finalization_requested[str(cafe_id)] = False
            current_occupancy[str(cafe_id)] = {}


def generate_yolo_frames(video_path, filename, cafe_id: str, camera_id=None):
    from ultralytics import YOLO

    logger.info("Starting stream for %s", filename)
    print(f"▶️ STREAM STARTED: {filename}")
    _ensure_live_session_dirs(cafe_id)
    with live_session_lock:
        if active_streams.get(cafe_id, 0) == 0:
            live_session_track_times[str(cafe_id)] = {}
        active_streams[str(cafe_id)] = active_streams.get(cafe_id, 0) + 1
    model = YOLO("yolov8s.pt")
    processed_customer_ids = set()
    track_zone_stats = {}
    cafe_track_times = live_session_track_times.setdefault(str(cafe_id), {})
    analytics_zones = _expand_zone_bounds_for_seating(_load_floor_plan_map_zones(cafe_id))
    last_zone_refresh_monotonic = time.monotonic()
    homography_status_logged = False
    logger.info("Opening video source: %s", video_path)
    capture = cv2.VideoCapture(os.fspath(video_path))
    if not capture.isOpened():
        logger.critical("Failed to open video source: %s", video_path)
    else:
        logger.info("Video source opened successfully: %s", video_path)
    capture.release()
    cafe = _get_cafe_by_id(cafe_id)
    floor_plan_path = _get_cafe_floor_plan_path(cafe)
    plan_img = cv2.imread(os.fspath(floor_plan_path)) if floor_plan_path is not None else None
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
        with live_session_lock:
            streaming_filenames[str(cafe_id)].add(filename)
        for r in results_generator:
            annotated_frame = r.orig_img.copy()
            projected_points: list[dict[str, object]] = []
            try:
                now_monotonic = time.monotonic()
                if now_monotonic - last_zone_refresh_monotonic >= 2.0:
                    analytics_zones = _expand_zone_bounds_for_seating(_load_floor_plan_map_zones(cafe_id))
                    last_zone_refresh_monotonic = now_monotonic
                current_time = datetime.now()
                height, width = annotated_frame.shape[:2]
                raw_zone = _load_zone_for_filename(cafe_id, filename)
                calibration = _load_calibration_for_filename(cafe_id, filename)
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
                                track_time = cafe_track_times.setdefault(
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
                                        "zone_name": "",
                                        "position_samples": [],
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
                            zone_name = _resolve_zone_name(map_x, map_y, analytics_zones)
                            projected_points.append(
                                {"x": map_x, "y": map_y, "is_staff": bool(is_staff), "zone_name": zone_name}
                            )
                            if live_track_id is not None:
                                with live_session_lock:
                                    track_time = cafe_track_times.setdefault(
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
                                            "zone_name": "",
                                            "position_samples": [],
                                        },
                                    )
                                    _append_position_sample(
                                        track_time,
                                        sample_time=current_time,
                                        x=map_x,
                                        y=map_y,
                                        zone_name=zone_name,
                                    )

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bg_color, 2)

                        if not is_staff and tid is not None:
                            with live_session_lock:
                                should_save_snapshot = not session_finalization_requested.get(str(cafe_id), False)

                            if should_save_snapshot and tid not in processed_customer_ids:
                                crop = _safe_crop_person(r.orig_img, (x1, y1, x2, y2))
                                if crop is not None:
                                    snapshot_path = _save_customer_snapshot(crop, cafe_id, camera_id, tid)
                                    if snapshot_path is not None:
                                        processed_customer_ids.add(tid)
                                        if live_track_id is not None:
                                            with live_session_lock:
                                                track_time = cafe_track_times.setdefault(
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
                                                        "zone_name": "",
                                                        "position_samples": [],
                                                    },
                                                )
                                                track_time["snapshots_saved"] = 1

                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw, y1), bg_color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            except Exception as exc:
                logger.error(f"Error processing frame: {exc}", exc_info=True)

            cafe_occupancy = current_occupancy.setdefault(str(cafe_id), {})
            cafe_occupancy[filename] = projected_points

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                logger.warning("JPEG encoding failed for %s", filename)
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        current_occupancy.setdefault(str(cafe_id), {})[filename] = []
        with live_session_lock:
            streaming_filenames[str(cafe_id)].discard(filename)
            _ensure_live_session_dirs(cafe_id)
            _write_live_people_times_csv(
                _get_cafe_live_snapshots_dir(cafe_id) / "people_times.csv",
                cafe_track_times,
            )
            active_streams[str(cafe_id)] = max(0, active_streams.get(str(cafe_id), 0) - 1)

@app.post("/api/auth/register")
async def register_user(user: UserRegister):
    email = _normalize_email(user.email)
    full_name = str(user.full_name or "").strip()
    password = str(user.password or "")
    cafe_name = str(user.cafe_name or "").strip()
    cafe_address = str(user.cafe_address or "").strip()

    if not full_name or not email or not password or not cafe_name or not cafe_address:
        raise HTTPException(status_code=400, detail="Усі поля є обов'язковими.")

    existing_user = users_collection.find_one({"email": email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Користувач з такою email адресою вже існує.")

    hashed_password = pwd_context.hash(password)
    result = users_collection.insert_one({
        "full_name": full_name,
        "email": email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
    })
    user_id = result.inserted_id

    new_cafe = {
        "name": cafe_name,
        "address": cafe_address,
        "owner_id": str(user_id),
        "floor_plan_url": "",
        "created_at": datetime.utcnow(),
    }
    cafes_collection.insert_one(new_cafe)

    return {"message": "Користувача успішно зареєстровано."}


@app.post("/api/auth/login", response_model=Token)
async def login_user(user: UserLogin):
    email = _normalize_email(user.email)
    password = str(user.password or "")

    if not email or not password:
        raise HTTPException(status_code=401, detail="Невірний email або пароль.")

    db_user = users_collection.find_one({"email": email})
    if not db_user or not pwd_context.verify(password, str(db_user.get("hashed_password", ""))):
        raise HTTPException(status_code=401, detail="Невірний email або пароль.")

    token = create_access_token({"sub": email})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/api/auth/me")
async def read_users_me(current_user=Depends(get_current_user)):
    return {
        "full_name": current_user.get("full_name"),
        "email": current_user.get("email"),
    }


@app.get("/api/cafes/me")
async def read_my_cafe(current_user=Depends(get_current_user)):
    user_cafe = cafes_collection.find_one({"owner_id": str(current_user["_id"])})
    if not user_cafe:
        raise HTTPException(status_code=404, detail="Cafe not found")

    return {
        "name": user_cafe.get("name", ""),
        "address": user_cafe.get("address", ""),
        "floor_plan_url": user_cafe.get("floor_plan_url", ""),
    }


async def _save_cafe_floor_plan_upload(user_cafe: dict, file: UploadFile) -> str:
    """Persist a new floor-plan file with a unique name (cache bust) and store a relative URL."""
    cafe_id = str(user_cafe["_id"])
    raw_name = file.filename or "plan.png"
    ext = raw_name.split(".")[-1].lower() if "." in raw_name else "png"
    if not ext.isalnum() or len(ext) > 8:
        ext = "png"
    new_filename = f"cafe_{cafe_id}_plan_{uuid4().hex[:8]}.{ext}"
    dest = CAFE_UPLOADS_DIR / new_filename

    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    await file.close()

    new_url = f"/uploads/{new_filename}"
    cafes_collection.update_one(
        {"_id": user_cafe["_id"]},
        {"$set": {"floor_plan_url": new_url}},
    )
    return new_url


@app.post("/api/cafe/update-floor-plan")
async def update_cafe_floor_plan(file: UploadFile = File(...), current_user=Depends(get_current_user)):
    user_cafe = cafes_collection.find_one({"owner_id": str(current_user["_id"])})
    if not user_cafe:
        raise HTTPException(status_code=404, detail="Cafe not found")
    new_url = await _save_cafe_floor_plan_upload(user_cafe, file)
    return {"message": "Plan updated", "floor_plan_url": new_url}


@app.post("/api/cafes/me/floor-plan")
async def upload_my_cafe_floor_plan(file: UploadFile = File(...), current_user=Depends(get_current_user)):
    user_cafe = cafes_collection.find_one({"owner_id": str(current_user["_id"])})
    if not user_cafe:
        raise HTTPException(status_code=404, detail="Cafe not found")

    new_url = await _save_cafe_floor_plan_upload(user_cafe, file)
    return {"floor_plan_url": new_url, "message": "Plan updated"}


@app.post("/set_zone/{filename}")
async def set_zone(filename: str, request: Request, current_user=Depends(get_current_user)):
    points = await request.json()
    user_cafe = _get_cafe_for_user(current_user)
    _save_zone_for_filename(str(user_cafe["_id"]), filename, points)

    return {"status": "success"}


@app.get("/api/floor_plan/zones")
async def get_floor_plan_zones(current_user=Depends(get_current_user)):
    user_cafe = _get_cafe_for_user(current_user)
    return {"zones": _expand_zone_bounds_for_seating(_load_floor_plan_map_zones(str(user_cafe["_id"])))}


@app.post("/api/floor_plan/zones")
async def save_floor_plan_zones(payload: list[FloorPlanZone], current_user=Depends(get_current_user)):
    user_cafe = _get_cafe_for_user(current_user)
    saved_zones = _save_floor_plan_map_zones(str(user_cafe["_id"]), [zone.model_dump() for zone in payload])
    return {"status": "success", "zones": _expand_zone_bounds_for_seating(saved_zones)}


@app.post("/api/analyze_floor_plan")
async def analyze_floor_plan(file: UploadFile | None = File(None), current_user=Depends(get_current_user)):
    user_cafe = _get_cafe_for_user(current_user)
    if file is not None:
        payload = await file.read()
        await file.close()
    else:
        floor_plan_path = _get_cafe_floor_plan_path(user_cafe)
        if floor_plan_path is None:
            raise HTTPException(status_code=404, detail="Floor plan not found")
        payload = floor_plan_path.read_bytes()

    if not payload:
        raise HTTPException(status_code=400, detail="Floor plan image is empty")

    image_array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode floor plan image")

    zones = _detect_floor_plan_zones(image)
    return {"zones": zones}


@app.post("/calibrate_camera/{filename}")
async def calibrate_camera(filename: str, payload: CalibrationPayload, current_user=Depends(get_current_user)):
    user_cafe = _get_cafe_for_user(current_user)
    _save_calibration_for_filename(str(user_cafe["_id"]), filename, payload.model_dump())
    return {"status": "success"}


@app.get("/calibration/{filename}")
async def get_calibration(filename: str, current_user=Depends(get_current_user)):
    user_cafe = _get_cafe_for_user(current_user)
    calibration = _load_calibration_for_filename(str(user_cafe["_id"]), filename)
    if calibration is None:
        return {"src_pts": [], "dst_pts": []}
    return calibration

@app.post("/upload_video")
async def upload_video(
    request: Request,
    video: UploadFile = File(...),
    camera_id: int | None = Form(None),
    current_user=Depends(get_current_user),
):
    user_cafe = _get_cafe_for_user(current_user)
    unique_filename = _build_unique_filename(video.filename)
    file_path = UPLOAD_DIR / unique_filename
    with file_path.open("wb") as buffer:
        buffer.write(await video.read())
    await video.close()

    thumbnail_name = _save_video_thumbnail(file_path)
    
    # Додаємо camera_id до URL стріму, щоб бекенд знав, чию зону вантажити
    base_stream_url = str(request.url_for("video_feed", filename=unique_filename))
    auth_header = request.headers.get("authorization", "")
    bearer_token = auth_header.split(" ", 1)[1] if auth_header.lower().startswith("bearer ") else ""
    query_params = []
    if camera_id:
        query_params.append(f"camera_id={camera_id}")
    if bearer_token:
        query_params.append(f"token={bearer_token}")
    stream_url = f"{base_stream_url}?{'&'.join(query_params)}" if query_params else base_stream_url
    
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


@app.get("/api/live_video_streams")
async def live_video_streams(current_user=Depends(get_current_user)):
    """Filenames whose YOLO/MJPEG pipeline is still running for this cafe."""
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])
    with live_session_lock:
        active = sorted(streaming_filenames[cafe_id])
    return {"active_filenames": active}


@app.get("/api/video_source/{filename}")
async def api_video_source(filename: str, token: str = ""):
    """Serve uploaded source video for <video> playback (token query, same pattern as video_feed)."""
    current_user = _get_user_from_token(token)
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    user_cafe = _get_cafe_for_user(current_user)
    _ = user_cafe
    safe_name = os.path.basename(filename)
    path = UPLOAD_DIR / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path)


@app.post("/api/end_session")
async def end_session(background_tasks: BackgroundTasks, current_user=Depends(get_current_user)):
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with live_session_lock:
        _ensure_live_session_dirs(cafe_id)
        snapshots_dir = _get_cafe_live_snapshots_dir(cafe_id)
        has_snapshots = any(path.is_dir() for path in snapshots_dir.iterdir())
        if session_finalization_requested.get(cafe_id, False):
            return {"status": "already_processing"}
        if not has_snapshots:
            return {"status": "no_snapshots"}
        session_finalization_requested[cafe_id] = True

    background_tasks.add_task(_finalize_live_session, session_id, cafe_id)
    return {
        "status": "success",
        "message": "Session finalization started in background",
        "session_id": session_id,
    }


@app.get("/api/stats")
async def get_stats(
    request: Request,
    tab: Optional[str] = None,
    date: Optional[str] = None,
    time: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])
    age_order = ["0-17", "18-25", "26-35", "36-45", "46-55", "56+", "Unknown"]
    empty_group = {
        "total_customers": 0,
        "gender_distribution": [],
        "age_groups": [{"label": label, "count": 0} for label in age_order],
        "average_duration_sec": 0,
        "peak_hour": "Unknown",
    }
    floor_plan_url = user_cafe.get("floor_plan_url") or None

    try:
        sessions = list(sessions_collection.find({"cafe_id": cafe_id}, {"_id": 0, "session_id": 1, "faces": 1}))
    except Exception as exc:
        logger.warning("Could not load analytics stats: %s", exc)
        return {
            "overall": empty_group,
            "by_date": {},
            "zone_stats": [],
            "heatmap_points": [],
            "floor_plan_url": floor_plan_url,
        }

    if not sessions:
        return {
            "overall": empty_group,
            "by_date": {},
            "zone_stats": [],
            "heatmap_points": [],
            "floor_plan_url": floor_plan_url,
        }

    overall_rows: list[dict] = []
    rows_by_date: dict[str, list[dict]] = {}
    heatmap_points: list[dict[str, object]] = []
    zone_person_samples: dict[str, dict[str, int]] = {}
    total_zone_samples = 0
    today_date = datetime.now().date()
    filter_date: Optional[datetime.date] = None
    range_start_date: Optional[datetime.date] = None
    start_time_value = None
    end_time_value = None

    if date:
        try:
            filter_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            filter_date = None
    elif tab == "today":
        filter_date = today_date
    elif tab == "7days":
        range_start_date = today_date - timedelta(days=7)
    elif tab == "30days":
        range_start_date = today_date - timedelta(days=30)

    if time and "-" in time:
        try:
            raw_start_time, raw_end_time = [part.strip() for part in time.split("-", 1)]
            start_time_value = datetime.strptime(raw_start_time, "%H:%M").time()
            end_time_value = datetime.strptime(raw_end_time, "%H:%M").time()
        except ValueError:
            start_time_value = None
            end_time_value = None

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
            enter_dt = _parse_sample_timestamp(enter_time)
            if not _matches_stats_filters(
                enter_dt,
                filter_date=filter_date,
                range_start_date=range_start_date,
                start_time_value=start_time_value,
                end_time_value=end_time_value,
            ):
                continue

            date_key = enter_time[:10] if len(enter_time) >= 10 else "Unknown Date"
            hour_str = enter_time[11:13] if len(enter_time) >= 13 else "Unknown"
            person_key = str(face.get("person_key") or face.get("face_id") or "")

            face_row = {
                "age": face.get("classification_result", {}).get("age"),
                "gender": face.get("classification_result", {}).get("sex"),
                "duration_sec": additional_metadata.get("duration_sec", 0),
                "hour": hour_str,
            }
            overall_rows.append(face_row)
            rows_by_date.setdefault(date_key, []).append(face_row)

            raw_samples = additional_metadata.get("position_samples", [])
            samples = raw_samples if isinstance(raw_samples, list) else []
            for sample in samples:
                if not isinstance(sample, dict):
                    continue

                sample_dt = _parse_sample_timestamp(sample.get("timestamp"))
                if not _matches_stats_filters(
                    sample_dt,
                    filter_date=filter_date,
                    range_start_date=range_start_date,
                    start_time_value=start_time_value,
                    end_time_value=end_time_value,
                ):
                    continue

                try:
                    sample_x = int(round(float(sample.get("x"))))
                    sample_y = int(round(float(sample.get("y"))))
                except (TypeError, ValueError):
                    continue

                heatmap_points.append({"x": sample_x, "y": sample_y, "weight": 1})

                sample_zone_name = str(sample.get("zone_name", "") or "").strip()
                if not sample_zone_name:
                    continue

                zone_counts = zone_person_samples.setdefault(sample_zone_name, {})
                zone_counts[person_key] = zone_counts.get(person_key, 0) + 1
                total_zone_samples += 1

    zone_stats = []
    for zone_name, person_counts in sorted(
        zone_person_samples.items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    ):
        total_samples = sum(person_counts.values())
        visitors_in_zone = max(1, len(person_counts))
        zone_stats.append(
            {
                "name": zone_name,
                "avg_time_sec": round((total_samples * POSITION_SAMPLE_INTERVAL_SEC) / visitors_in_zone),
                "percent": round((total_samples / total_zone_samples) * 100) if total_zone_samples > 0 else 0,
            }
        )

    if not overall_rows:
        return {
            "overall": empty_group,
            "by_date": {},
            "zone_stats": zone_stats,
            "heatmap_points": heatmap_points,
            "floor_plan_url": floor_plan_url,
        }

    return {
        "overall": _build_stats_group(overall_rows, age_order),
        "by_date": {
            date_key: _build_stats_group(date_rows, age_order)
            for date_key, date_rows in sorted(rows_by_date.items(), key=lambda item: item[0], reverse=True)
        },
        "zone_stats": zone_stats,
        "heatmap_points": heatmap_points,
        "floor_plan_url": floor_plan_url,
    }


@app.get("/api/analytics/summary")
async def get_analytics_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_range: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])

    period_start, period_end = _require_analytics_period_or_all(start_date, end_date)
    start_time_value, end_time_value = _parse_query_time_range_param(time_range)

    sessions = _sessions_for_cafe(cafe_id)

    cur_n, cur_avg, peak_hour, peak_clients = _aggregate_visitor_faces_for_period(
        sessions,
        period_start=period_start,
        period_end=period_end,
        start_time_value=start_time_value,
        end_time_value=end_time_value,
    )

    clients_trend: float | None = None
    duration_trend: float | None = None

    if period_start is not None and period_end is not None:
        span_days = (period_end - period_start).days + 1
        prev_end = period_start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=span_days - 1)
        prev_n, prev_avg, _, _ = _aggregate_visitor_faces_for_period(
            sessions,
            period_start=prev_start,
            period_end=prev_end,
            start_time_value=start_time_value,
            end_time_value=end_time_value,
        )
        clients_trend = _percent_change(float(cur_n), float(prev_n))
        duration_trend = _percent_change(cur_avg, prev_avg)

    return {
        "total_clients": int(cur_n),
        "clients_trend": clients_trend,
        "avg_duration_seconds": round(cur_avg, 2),
        "duration_trend": duration_trend,
        "peak_hour": peak_hour,
        "peak_clients": peak_clients,
    }


@app.get("/api/analytics/dynamics")
async def get_analytics_dynamics(
    view: str = "hours",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_range: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    """
    Visit dynamics for the authenticated owner's cafe only.
    Sessions store per-face enter_time (not a top-level session timestamp); filtering uses that datetime.
    """
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])
    period_start, period_end = _require_analytics_period_or_all(start_date, end_date)
    st_t, en_t = _parse_query_time_range_param(time_range)
    sessions = _sessions_for_cafe(cafe_id)
    rows = _visitor_face_rows(
        sessions,
        period_start=period_start,
        period_end=period_end,
        start_time_value=st_t,
        end_time_value=en_t,
    )
    safe_view = "days" if view == "days" else "hours"
    return _analytics_dynamics_series(rows, view=safe_view)


@app.get("/api/analytics/demographics")
async def get_analytics_demographics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_range: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])
    period_start, period_end = _require_analytics_period_or_all(start_date, end_date)
    st_t, en_t = _parse_query_time_range_param(time_range)
    sessions = _sessions_for_cafe(cafe_id)
    rows = _visitor_face_rows(
        sessions,
        period_start=period_start,
        period_end=period_end,
        start_time_value=st_t,
        end_time_value=en_t,
    )
    return _analytics_demographics_from_rows(rows)


@app.get("/api/analytics/history")
async def get_analytics_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_range: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])
    period_start, period_end = _require_analytics_period_or_all(start_date, end_date)
    st_t, en_t = _parse_query_time_range_param(time_range)
    sessions = _sessions_for_cafe(cafe_id)
    rows = _visitor_face_rows(
        sessions,
        period_start=period_start,
        period_end=period_end,
        start_time_value=st_t,
        end_time_value=en_t,
    )
    return _analytics_daily_table(rows)


@app.get("/api/analytics/daily-stats")
async def get_analytics_daily_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_range: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    """Alias for /api/analytics/history — per-day aggregates for the cafe."""
    return await get_analytics_history(start_date, end_date, time_range, current_user)


@app.get("/api/analytics/zones")
async def get_analytics_zones(current_user=Depends(get_current_user)):
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])
    palette = ["#8a895f", "#846851", "#411e03", "#d7cdc1", "#cbd5e1", "#94a3b8", "#d6d3d1"]

    zone_pipeline = [
        {"$match": {"cafe_id": cafe_id}},
        {"$unwind": "$faces"},
        {
            "$project": {
                "zone_name": {"$trim": {"input": {"$ifNull": ["$faces.additional_metadata.zone_name", ""]}}},
                "duration_sec": {
                    "$convert": {
                        "input": {
                            "$ifNull": [
                                "$faces.total_duration_sec",
                                "$faces.additional_metadata.duration_sec",
                            ]
                        },
                        "to": "double",
                        "onError": 0,
                        "onNull": 0,
                    }
                },
            }
        },
        {
            "$addFields": {
                "normalized_zone": {
                    "$cond": [
                        {"$eq": [{"$ifNull": ["$zone_name", ""]}, ""]},
                        "В русі / Інше",
                        "$zone_name",
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": "$normalized_zone",
                "visits": {"$sum": 1},
                "average_duration_sec": {"$avg": "$duration_sec"},
            }
        },
        {"$sort": {"visits": -1, "_id": 1}},
    ]

    raw_zones = list(sessions_collection.aggregate(zone_pipeline))
    total_visits = sum(int(zone.get("visits", 0) or 0) for zone in raw_zones)

    formatted_zones = []
    for index, zone in enumerate(raw_zones):
        visits = int(zone.get("visits", 0) or 0)
        percentage = round((visits / total_visits) * 100) if total_visits > 0 else 0
        formatted_zones.append(
            {
                "name": str(zone.get("_id", "") or "Unknown zone"),
                "avg_time": _format_duration_label(zone.get("average_duration_sec", 0)),
                "visits": visits,
                "percentage": int(percentage),
                "color": palette[index % len(palette)],
            }
        )

    return formatted_zones


@app.get("/api/analytics/routes")
async def get_analytics_routes(current_user=Depends(get_current_user)):
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])
    route_counts: defaultdict[str, int] = defaultdict(int)

    sessions = list(sessions_collection.find({"cafe_id": cafe_id}, {"_id": 0, "faces": 1}))
    for session in sessions:
        faces = session.get("faces", [])
        if not isinstance(faces, list):
            continue

        for face in faces:
            if not isinstance(face, dict):
                continue

            additional_metadata = face.get("additional_metadata", {}) or {}
            raw_samples = additional_metadata.get("position_samples", [])
            if not isinstance(raw_samples, list):
                continue

            sorted_samples = sorted(
                [sample for sample in raw_samples if isinstance(sample, dict)],
                key=lambda sample: (
                    _parse_sample_timestamp(sample.get("timestamp")) is None,
                    _parse_sample_timestamp(sample.get("timestamp")) or datetime.max,
                ),
            )

            compressed_sequence: list[str] = []
            previous_zone = ""
            for sample in sorted_samples:
                zone_name = str(sample.get("zone_name", "") or "").strip()
                if not zone_name or zone_name == previous_zone:
                    continue
                compressed_sequence.append(zone_name)
                previous_zone = zone_name

            if not compressed_sequence:
                continue

            route_path = " → ".join(compressed_sequence)
            route_counts[route_path] += 1

    total_valid_routes = sum(route_counts.values())
    if total_valid_routes <= 0:
        return []

    top_routes = sorted(route_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    return [
        {
            "path": route_path,
            "percentage": int(round((count / total_valid_routes) * 100)),
        }
        for route_path, count in top_routes
    ]


@app.get("/api/analytics/heatmap")
async def get_analytics_heatmap(current_user=Depends(get_current_user)):
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])

    heatmap_pipeline = [
        {"$match": {"cafe_id": cafe_id}},
        {"$unwind": "$faces"},
        {
            "$project": {
                "position_samples": {
                    "$ifNull": [
                        "$faces.position_samples",
                        "$faces.additional_metadata.position_samples",
                    ]
                }
            }
        },
        {"$unwind": "$position_samples"},
        {
            "$project": {
                "x": {
                    "$convert": {
                        "input": "$position_samples.x",
                        "to": "double",
                        "onError": -1,
                        "onNull": -1,
                    }
                },
                "y": {
                    "$convert": {
                        "input": "$position_samples.y",
                        "to": "double",
                        "onError": -1,
                        "onNull": -1,
                    }
                },
            }
        },
        {
            "$match": {
                "x": {"$gte": 0, "$lte": 3000},
                "y": {"$gte": 0, "$lte": 3000},
            }
        },
        {"$sample": {"size": 1000}},
        {"$project": {"_id": 0, "x": 1, "y": 1}},
    ]

    heatmap_points = list(sessions_collection.aggregate(heatmap_pipeline))
    return [
        {
            "x": int(round(float(point.get("x", 0) or 0))),
            "y": int(round(float(point.get("y", 0) or 0))),
        }
        for point in heatmap_points
    ]


@app.get("/api/occupancy")
async def get_occupancy(current_user=Depends(get_current_user)):
    user_cafe = _get_cafe_for_user(current_user)
    cafe_id = str(user_cafe["_id"])
    safe_occupancy = {
        camera_filename: points if isinstance(points, list) else []
        for camera_filename, points in current_occupancy.get(cafe_id, {}).items()
    }
    merged_points = _merge_projected_points(safe_occupancy)
    return {"total_customers": len(merged_points), "points": merged_points}

@app.get("/video_feed/{filename}", name="video_feed")
async def video_feed(filename: str, camera_id: int = None, token: str = ""):
    current_user = _get_user_from_token(token)
    user_cafe = _get_cafe_for_user(current_user)
    video_path = UPLOAD_DIR / filename
    return StreamingResponse(
        generate_yolo_frames(str(video_path), filename, str(user_cafe["_id"]), camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)