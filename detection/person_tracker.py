from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import random
import logging
from snapshot_manager import SnapshotManager
import cv2

class PersonTracker:
    def __init__(self, model_path="yolov8n.pt", class_names=["person"], min_conf=0.4,
                 min_box_size=10, alpha=0.3, process_every_n=2, snaphots_path=None):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.min_conf = min_conf
        self.min_box_size = min_box_size
        self.alpha = alpha
        self.process_every_n = process_every_n

        self.last_positions = {}
        self.last_tracks = []
        self.track_colors = {}
        self.track_history = defaultdict(list)
        self.times = {}
        self.saved_photos = defaultdict(list)
        self.snaphots_path = snaphots_path

    def process_frame(self, frame, orig_frame):
        active_ids = set()

        results = self.model.track(
            frame,
            persist=True,
            imgsz=640,
            tracker="botsort.yaml",
            conf=self.min_conf,
            iou=0.5
        )
        self.last_tracks = results[0].boxes

        now = datetime.now()
        for box in self.last_tracks or []:
            if box.id is None:
                continue
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            if cls_name not in self.class_names or conf < self.min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < self.min_box_size or h < self.min_box_size:
                continue

            track_id = int(box.id[0])
            active_ids.add(track_id)

            if track_id not in self.track_colors:
                self.track_colors[track_id] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )

            # Timing
            if track_id not in self.times:
                self.times[track_id] = {"enter": now, "last": now, "exit": None, "duration": None}
                logging.info(f"Person {track_id} entered at {now.strftime('%H:%M:%S')}")
            else:
                self.times[track_id]["last"] = now

            # Smooth positions
            if track_id in self.last_positions:
                px1, py1, px2, py2 = self.last_positions[track_id]
                x1 = int(px1 + self.alpha * (x1 - px1))
                y1 = int(py1 + self.alpha * (y1 - py1))
                x2 = int(px2 + self.alpha * (x2 - px2))
                y2 = int(py2 + self.alpha * (y2 - py2))

            self.last_positions[track_id] = (x1, y1, x2, y2)
            color = self.track_colors[track_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            delta = (now - self.times[track_id]["enter"]).total_seconds()
            cv2.putText(frame, f"ID {track_id} {delta:.1f}s", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
            
            # Snapshot
            SnapshotManager.save_snapshot(orig_frame, track_id, (x1, y1, x2, y2),
                                          self.saved_photos, self.snaphots_path)

        return frame
