import cv2
import numpy as np
import os
import time
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
import random
from ultralytics.utils import LOGGER
import logging
import csv
import pyautogui

# Settings
LOGGER.setLevel(40)
logging.basicConfig(
    filename="logs/people_tracking.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class VideoProcessor:
    def __init__(self, video_path, output_path, use_screen=False):
        self.use_screen = use_screen
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Tracking", 960, 540)
        self.frame_idx = 0

    def read_frame(self):
        self.frame_idx += 1
        if self.use_screen:
            screenshot = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
        return frame

    def show_and_write(self, frame):
        cv2.imshow("YOLOv8 Tracking", frame)
        self.out.write(frame)

    def release(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


class PersonTracker:
    def __init__(self, model_path="yolov8n.pt", class_names=["person"], min_conf=0.4,
                 min_box_size=10, alpha=0.3, process_every_n=2, snaphots_path = None):
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
            
<<<<<<< HEAD
            # Snapshot
            SnapshotManager.save_snapshot(orig_frame, track_id, (x1, y1, x2, y2), self.saved_photos)
=======
            # --- Snapshot ---
            SnapshotManager.save_snapshot(orig_frame, track_id, (x1, y1, x2, y2), self.saved_photos, self.snaphots_path)
>>>>>>> 27a929fb9c11e2e39932205023698c82506c43de

        return frame


class SnapshotManager:
    snapshots_dir = "snapshots"

    @staticmethod
    def save_snapshot(orig_frame, track_id, bbox, saved_photos, snaphots_path):
        snaphots_path = snaphots_path if snaphots_path else SnapshotManager.snapshots_dir
        
        x1, y1, x2, y2 = bbox
        person_dir = os.path.join(snaphots_path, str(track_id))
        os.makedirs(person_dir, exist_ok=True)

        if len(saved_photos[track_id]) < 5:
            crop = orig_frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0:
                filename = f"{int(time.time()*1000)}.jpg"
                filepath = os.path.join(person_dir, filename)
                cv2.imwrite(filepath, crop)
                saved_photos[track_id].append(filepath)
                logging.debug(f"Saved snapshot for ID {track_id}: {filepath}")


class StatisticsLogger:
    @staticmethod
    def finalize(times):
        now = datetime.now()
        for tid, t in times.items():
            if t["exit"] is None:
                t["exit"] = now
                t["duration"] = (t["exit"] - t["enter"]).total_seconds()
                logging.info(f"Person {tid} still in frame at end (duration {t['duration']:.1f}s)")

        total_visitors = len(times)
        if total_visitors > 0:
            durations = [t['duration'] for t in times.values() if t['duration']]
            logging.info(f"Total visitors detected: {total_visitors}")
            logging.info(f"Average stay duration: {sum(durations)/total_visitors:.1f}s")
            logging.info(f"Max stay duration: {max(durations):.1f}s")
            logging.info(f"Min stay duration: {min(durations):.1f}s")
        else:
            logging.info("No visitors detected")

        # Save CSV
        with open("logs/people_times.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Enter", "Exit", "Duration_sec"])
            for tid, t in times.items():
                writer.writerow([
                    tid,
                    t["enter"].strftime("%Y-%m-%d %H:%M:%S") if t["enter"] else "",
                    t["exit"].strftime("%Y-%m-%d %H:%M:%S") if t["exit"] else "",
                    f"{t['duration']:.1f}" if t["duration"] else ""
                ])


class DetectionManager:
    def __init__(self, video_path, output_path, snaphots_path):
        self.processor = VideoProcessor(video_path, output_path)
        self.tracker = PersonTracker(snaphots_path=snaphots_path)

    def run(self):
        while True:
            frame = self.processor.read_frame()
            if frame is None:
                break
            orig_frame = frame.copy()
            processed_frame = self.tracker.process_frame(frame, orig_frame)
            self.processor.show_and_write(processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.processor.release()
        StatisticsLogger.finalize(self.tracker.times)


if __name__ == "__main__":
    app = DetectionManager("coffee_shop.mp4", "processed_video.mp4", None)
    app.run()
