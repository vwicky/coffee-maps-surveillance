import csv
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from app_paths import LOGS_DIR, ensure_data_dirs


def _load_people_times_csv(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}

    people: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            person_id = str(row.get("ID", "")).strip()
            if not person_id:
                continue
            people[person_id] = row
    return people


def _load_faces_by_track_id(mongo_manager, session_id: str) -> Dict[str, Dict[str, Any]]:
    faces = mongo_manager.get_faces(session_id) or []
    by_track_id: Dict[str, Dict[str, Any]] = {}

    for face in faces:
        person_key = face.get("person_key")
        if person_key is None:
            additional = face.get("additional_metadata") or {}
            person_key = additional.get("frame_number")
        if person_key is None:
            continue

        key = str(person_key)
        if key in by_track_id:
            continue
        by_track_id[key] = face

    return by_track_id


def build_people_final_report(
    mongo_manager,
    session_id: str,
    people_times_csv_path: str = str(LOGS_DIR / "people_times.csv"),
    output_csv_path: str = str(LOGS_DIR / "people_final_report.csv"),
) -> str:
    times_data = _load_people_times_csv(people_times_csv_path)
    faces_by_track_id = _load_faces_by_track_id(mongo_manager, session_id)

    matched_track_ids = set()
    for face in faces_by_track_id.values():
        for tid in (face.get("track_ids") or []):
            matched_track_ids.add(str(tid))
        additional = face.get("additional_metadata") or {}
        frame_num = additional.get("frame_number")
        if frame_num is not None:
            matched_track_ids.add(str(frame_num))

    all_ids = sorted(
        set(faces_by_track_id.keys()) | {tid for tid in times_data.keys() if tid not in matched_track_ids},
        key=lambda x: int(x) if x.isdigit() else x,
    )

    ensure_data_dirs()
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID",
            "Enter",
            "Exit",
            "Duration_sec",
            "Frames_seen",
            "Max_conf",
            "Age",
            "Gender",
            "Face_id",
            "Embedding_id",
        ])

        for person_id in all_ids:
            time_row = times_data.get(person_id, {})
            face_row = faces_by_track_id.get(person_id, {})
            cl = face_row.get("classification_result") or {}
            additional = face_row.get("additional_metadata") or {}
            total_duration = face_row.get("total_duration_sec")
            if total_duration is not None:
                duration_out = f"{float(total_duration):.1f}"
            else:
                duration_out = time_row.get("Duration_sec", "")
            enter_out = time_row.get("Enter", "") or additional.get("enter_time", "")
            exit_out = time_row.get("Exit", "") or additional.get("exit_time", "")

            writer.writerow([
                person_id,
                enter_out,
                exit_out,
                duration_out,
                time_row.get("Frames_seen", ""),
                time_row.get("Max_conf", ""),
                cl.get("age", ""),
                cl.get("sex", ""),
                face_row.get("face_id", ""),
                face_row.get("embedding_id", ""),
            ])

    return str(output_path)


def build_visitors_json_report(
    merged_csv_path: str = str(LOGS_DIR / "people_final_report.csv"),
    output_json_path: str = str(LOGS_DIR / "visitors_report.json"),
    staff_ids: set[str] | None = None,
) -> str:
    staff_ids = staff_ids or set()
    rows: list[Dict[str, Any]] = []

    if not os.path.exists(merged_csv_path):
        ensure_data_dirs()
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return str(output_path)

    with open(merged_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            person_id = str(row.get("ID", "")).strip()
            if not person_id:
                continue

            enter_raw = (row.get("Enter") or "").strip()
            exit_raw = (row.get("Exit") or "").strip()

            enter_dt = None
            exit_dt = None
            try:
                if enter_raw:
                    enter_dt = datetime.strptime(enter_raw, "%Y-%m-%d %H:%M:%S")
                if exit_raw:
                    exit_dt = datetime.strptime(exit_raw, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # Keep values as-is if format differs unexpectedly.
                pass

            duration_sec_raw = (row.get("Duration_sec") or "").strip()
            try:
                duration_minutes = round(float(duration_sec_raw) / 60.0, 2) if duration_sec_raw else None
            except ValueError:
                duration_minutes = None

            age_raw = (row.get("Age") or "").strip()
            age_value = None
            if age_raw:
                try:
                    age_value = int(float(age_raw))
                except ValueError:
                    age_value = None

            visitor = {
                "visitor_id": f"unique_id_{person_id.zfill(3)}",
                "date": enter_dt.strftime("%Y-%m-%d") if enter_dt else (enter_raw.split(" ")[0] if enter_raw else ""),
                "enter_time": enter_dt.strftime("%H:%M:%S") if enter_dt else (enter_raw.split(" ")[-1] if enter_raw else ""),
                "exit_time": exit_dt.strftime("%H:%M:%S") if exit_dt else (exit_raw.split(" ")[-1] if exit_raw else ""),
                "duration_minutes": duration_minutes,
                "demographics": {
                    "age": age_value,
                    "sex": (row.get("Gender") or "").strip(),
                },
                "category": "staff" if person_id in staff_ids else "customer",
            }
            rows.append(visitor)

    ensure_data_dirs()
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return str(output_path)
