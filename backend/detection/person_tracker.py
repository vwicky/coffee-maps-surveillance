from ultralytics import YOLO
import cv2
from datetime import datetime
import collections
import numpy as np
import logging
import os

# НАШІ МОДУЛІ (Переконайтеся, що шляхи правильні)
from app_paths import LOGS_DIR, SNAPSHOTS_DIR, ZONES_JSON_PATH
from classification.src.staff_detector import StaffDetector
from .snapshot_manager import SnapshotManager
from .tracking_validation import TrackingValidationLogger

class PersonTracker:
    def __init__(self, model_path="yolov8l.pt", class_names=["person"], 
                 min_conf=0.4,          # Підвищили суворість до 60%
                 min_area=8000,        # Відсікаємо дрібні об'єкти (комах)
                 video_fps=30,          # Швидкість вашого відео (зазвичай 30)      
                 snaphots_path=os.fspath(SNAPSHOTS_DIR),
                 smoothing_alpha=0.4,  # 0..1 (вище = швидше реагує, нижче = стабільніше)
                 max_missing_seconds=0.8, # як довго тримати рамку при короткій втраті детекції
                 # Minimum *video* time (from frame count / fps) before saving snapshots.
                 # Wall-clock delta breaks offline processing (fast GPU ≠ video seconds).
                 snapshot_min_video_seconds=0.25,
                 enable_validation=True,
                 validation_csv_path=os.fspath(LOGS_DIR / "tracking_validation.csv"),
                 validation_summary_path=os.fspath(LOGS_DIR / "tracking_validation_summary.json"),
                 ):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.min_conf = min_conf
        self.min_area = min_area
        self.snaphots_path = snaphots_path
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_missing_seconds = float(max_missing_seconds)
        self.video_fps = int(video_fps) if video_fps else 0
        self.snapshot_min_video_seconds = float(snapshot_min_video_seconds)

        self.last_positions = {}
        self._prev_active_ids: set = set()
        self.times = collections.defaultdict(dict)
        self.track_colors = collections.defaultdict(lambda: self._generate_color())
        self.saved_photos = collections.defaultdict(list) 
        self.log_every_seconds = 2.0
        self.frame_idx = 0
        self.staff_detector = StaffDetector(
            zones_json_path=os.fspath(ZONES_JSON_PATH),
            session_id="session_1",
            threshold=0.7,
        )

        self.validation = None
        if enable_validation:
            self.validation = TrackingValidationLogger(
                fps=self.video_fps,
                csv_path=os.fspath(validation_csv_path),
                summary_json_path=os.fspath(validation_summary_path),
            )
        self._validation_finished_ids = set()

    def _draw_staff_zone(self, frame) -> None:
        """Draw a subtle semi-transparent outline of the configured staff ROI."""
        zone = getattr(self.staff_detector, "staff_zone", None)
        if zone is None or len(zone) < 3:
            return

        overlay = frame.copy()
        cv2.polylines(
            overlay,
            [zone.reshape((-1, 1, 2))],
            isClosed=True,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # Blend only the ROI outline back onto the frame so the blue guide stays visible
        # without dominating the surveillance view.
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    def _effective_fps(self) -> float:
        return float(self.video_fps) if self.video_fps > 0 else 30.0

    def _video_dwell_seconds(self, track_id: int) -> float:
        info = self.times.get(track_id) or {}
        first_f = info.get("first_frame")
        if first_f is None:
            return 0.0
        return (self.frame_idx - int(first_f)) / self._effective_fps()

    def _save_snapshot_if_needed(self, orig_frame, track_id: int, bbox: tuple) -> None:
        SnapshotManager.save_snapshot(
            orig_frame,
            track_id,
            bbox,
            self.saved_photos,
            self.snaphots_path,
        )
        snapshot_info = self.saved_photos.get(track_id)
        if isinstance(snapshot_info, dict):
            self.times[track_id]["best_snapshot"] = snapshot_info.get("path")

    def finalize_pending_snapshots(self, orig_frame) -> None:
        """Last chance to write snapshots for tracks that never met the dwell threshold or are still active at EOF."""
        if orig_frame is None:
            return
        for track_id, info in list(self.times.items()):
            if isinstance(self.saved_photos.get(track_id), dict):
                continue
            bbox = info.get("last_bbox")
            if not bbox or info.get("frames_seen", 0) < 1:
                continue
            self._save_snapshot_if_needed(orig_frame, track_id, tuple(map(int, bbox)))

    def process_frame(self, frame, orig_frame):
        self.frame_idx += 1
        active_ids = set()
        drawn_ids = set()
        self._draw_staff_zone(frame)

        # Використовуємо стандартний стабільний bytetrack
        results = self.model.track(
            frame,
            persist=True,
            imgsz=640,
            tracker="bytetrack.yaml", 
            conf=self.min_conf,
            iou=0.5,
            verbose=False
        )
        self.last_tracks = results[0].boxes

        now = datetime.now()
        for box in self.last_tracks or []:
            if box.id is None:
                continue
            
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            
            # Фільтр: тільки люди і тільки висока впевненість
            if cls_name not in self.class_names or conf < self.min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            
            # Фільтр: відсікаємо дрібних "привидів" за площею
            area = w * h
            if area < self.min_area:
                continue

            track_id = int(box.id[0])
            active_ids.add(track_id)

            # Легка стабілізація bbox щоб рамка не "стрибала" на рухах/нахилах.
            if self.smoothing_alpha > 0 and track_id in self.last_positions:
                px1, py1, px2, py2 = self.last_positions[track_id]
                a = self.smoothing_alpha
                x1 = int(px1 + a * (x1 - px1))
                y1 = int(py1 + a * (y1 - py1))
                x2 = int(px2 + a * (x2 - px2))
                y2 = int(py2 + a * (y2 - py2))

            self.last_positions[track_id] = (x1, y1, x2, y2)
            bbox = [x1, y1, x2, y2]
            is_staff = self.staff_detector.update_and_check(track_id, bbox)
            
            # Обчислення часу перебування
            if track_id not in self.times:
                self.times[track_id] = {
                    "enter": now,
                    "last_seen": now,
                    "last_log_at": now,
                    "first_frame": self.frame_idx,
                    "first_bbox": [x1, y1, x2, y2],
                    "last_bbox": [x1, y1, x2, y2],
                    "max_area": area,
                    "frames_seen": 0,
                    "max_conf": conf,
                    "is_staff": is_staff,
                }
                logging.info(
                    "ENTER person=%s conf=%.3f bbox=(%d,%d,%d,%d) area=%d",
                    track_id, conf, x1, y1, x2, y2, area
                )
            
            self.times[track_id]["last_seen"] = now
            self.times[track_id]["last_bbox"] = [x1, y1, x2, y2]
            self.times[track_id]["frames_seen"] += 1
            self.times[track_id]["max_area"] = max(self.times[track_id]["max_area"], area)
            self.times[track_id]["max_conf"] = max(self.times[track_id]["max_conf"], conf)
            self.times[track_id]["is_staff"] = is_staff
            delta = (now - self.times[track_id]["enter"]).total_seconds()
            dwell_video = self._video_dwell_seconds(track_id)

            # Periodic status log to understand location and duration before replay.
            since_last_log = (now - self.times[track_id]["last_log_at"]).total_seconds()
            if since_last_log >= self.log_every_seconds:
                self.times[track_id]["last_log_at"] = now
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                logging.info(
                    "TRACK person=%s t=%.1fs center=(%d,%d) bbox=(%d,%d,%d,%d) conf=%.3f max_area=%d frames=%d",
                    track_id,
                    delta,
                    cx,
                    cy,
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                    self.times[track_id]["max_area"],
                    self.times[track_id]["frames_seen"],
                )
            
            # Колір та рамка
            color = (0, 0, 255) if is_staff else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            drawn_ids.add(track_id)
            
            # Текст: ID та Час перебування (video time matches file timeline when fps is correct)
            label_prefix = "Staff" if is_staff else f"ID {track_id}"
            label = f"{label_prefix} {dwell_video:.1f}s"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Snapshot: use *video* dwell so offline runs still match the recording; keep best (largest bbox) crop.
            if dwell_video >= self.snapshot_min_video_seconds:
                self._save_snapshot_if_needed(orig_frame, track_id, (x1, y1, x2, y2))

        # Tracks that disappeared this frame: save at least one snapshot if we never did (short visits).
        lost_ids = self._prev_active_ids - active_ids
        for lost_id in lost_ids:
            info = self.times.get(lost_id)
            if not info or isinstance(self.saved_photos.get(lost_id), dict):
                continue
            if info.get("frames_seen", 0) < 1:
                continue
            lb = info.get("last_bbox")
            if lb:
                self._save_snapshot_if_needed(orig_frame, lost_id, tuple(map(int, lb)))

        self._prev_active_ids = set(active_ids)

        # Якщо детекція пропала на 0.2-0.8с (нахил/рух/розмиття) — не прибираємо рамку одразу.
        if self.max_missing_seconds > 0:
            for track_id, info in list(self.times.items()):
                if track_id in active_ids:
                    continue
                last_seen = info.get("last_seen")
                last_bbox = info.get("last_bbox")
                if last_seen is None or not last_bbox:
                    continue
                missing_for = (now - last_seen).total_seconds()
                if missing_for <= self.max_missing_seconds:
                    x1, y1, x2, y2 = map(int, last_bbox)
                    is_staff = bool(info.get("is_staff", False))
                    color = (0, 0, 255) if is_staff else (0, 255, 0)
                    # Тонша рамка щоб було видно що трек "тимчасово втрачено"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    drawn_ids.add(track_id)
                    delta = (last_seen - info.get("enter", last_seen)).total_seconds()
                    label_prefix = "Staff" if is_staff else f"ID {track_id}"
                    cv2.putText(
                        frame,
                        f"{label_prefix} {delta:.1f}s",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        1
                    )

        # Validation: compare expected vs actual visibility per track per frame.
        if self.validation is not None:
            now = datetime.now()
            for track_id, info in list(self.times.items()):
                detected = track_id in active_ids
                actual_visible = track_id in drawn_ids
                last_seen = info.get("last_seen")
                missing_for_s = None
                if last_seen is not None:
                    missing_for_s = (now - last_seen).total_seconds()
                expected_visible = detected or (
                    missing_for_s is not None and missing_for_s <= self.max_missing_seconds
                )

                tid = int(track_id)
                # Once a track is no longer expected to be visible, stop emitting rows for it.
                # This avoids huge CSVs on long videos and keeps "gap" metrics meaningful.
                if not detected and not expected_visible and not actual_visible:
                    self._validation_finished_ids.add(tid)
                    continue

                # Rare: if tracker re-activates an ID, resume validation for it.
                if detected and tid in self._validation_finished_ids:
                    self._validation_finished_ids.discard(tid)

                self.validation.log_frame_for_track(
                    frame_idx=self.frame_idx,
                    track_id=tid,
                    detected=detected,
                    expected_visible=expected_visible,
                    actual_visible=actual_visible,
                    missing_for_s=missing_for_s,
                )

        return frame

    def finalize_validation(self):
        if self.validation is not None:
            self.validation.close()
            self.validation = None

    def _generate_color(self):
        # Повертає випадковий колір для треку (не червоний, не зелений)
        return (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))