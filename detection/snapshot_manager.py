import os
import time
import cv2
from collections import defaultdict
import logging

class SnapshotManager:
    snapshots_dir = "snapshots"

    @staticmethod
    def save_snapshot(orig_frame, track_id, bbox, saved_photos, snaphots_path=None):
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
