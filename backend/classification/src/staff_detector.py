from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class StaffDetector:
    """Classify tracked people as staff using their spatial behavior."""

    def __init__(self, zones_json_path: str, session_id: str, threshold: float = 0.7) -> None:
        self.zones_json_path = Path(zones_json_path)
        self.session_id = session_id
        self.threshold = threshold
        self.staff_zone = self._load_zone()
        # Format: {track_id: [total_frames_seen, frames_inside_zone]}
        self.track_stats: dict[int, list[int]] = {}

    def _load_zone(self) -> np.ndarray | None:
        """
        Load the polygon for the current session from a JSON file.

        Expected JSON format:
        {
            "session_1": [[400, 200], [800, 200], [900, 600], [300, 600]]
        }
        """
        try:
            with self.zones_json_path.open("r", encoding="utf-8") as file:
                zones: dict[str, Any] = json.load(file)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

        points = zones.get(self.session_id)
        if not points:
            return None

        zone = np.array(points, dtype=np.int32)
        if zone.ndim != 2 or zone.shape[1] != 2 or len(zone) < 3:
            return None

        return zone

    def is_inside(self, bbox: list[float] | tuple[float, float, float, float]) -> bool:
        """
        Check whether the person's feet point is inside the staff polygon.

        The feet point uses the bottom-center of the bounding box:
        x_center = (x1 + x2) / 2
        y_bottom = y2

        This is more stable than using the box center because the feet point
        better represents where the person is standing on the floor.
        """
        if self.staff_zone is None:
            return False

        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2.0
        y_bottom = y2
        feet_point = (float(x_center), float(y_bottom))

        # cv2.pointPolygonTest returns:
        #   1 if the point is inside
        #   0 if the point is exactly on the polygon edge
        #  -1 if the point is outside
        # Treating 0 as inside avoids flicker when the tracked point sits on
        # the boundary of the staff zone.
        return cv2.pointPolygonTest(self.staff_zone, feet_point, False) >= 0

    def update_and_check(
        self,
        track_id: int,
        bbox: list[float] | tuple[float, float, float, float],
        person_crop: np.ndarray | None = None,
    ) -> bool:
        """
        Update per-track statistics and return whether the person is staff.

        Threshold logic:
        - Count how many frames this track has been seen overall.
        - Count how many of those frames had the feet point inside the zone.
        - If inside_ratio = frames_inside_zone / total_frames_seen exceeds the
          threshold, classify the track as staff.
        """
        _ = person_crop
        if track_id not in self.track_stats:
            self.track_stats[track_id] = [0, 0]

        stats = self.track_stats[track_id]
        stats[0] += 1

        if self.is_inside(bbox):
            stats[1] += 1

        inside_ratio = stats[1] / stats[0]
        return inside_ratio > self.threshold

    def get_staff_ids(self) -> list[int]:
        """Return all track IDs whose inside-zone ratio exceeds the threshold."""
        staff_ids: list[int] = []

        for track_id, (total_frames_seen, frames_inside_zone) in self.track_stats.items():
            if total_frames_seen == 0:
                continue

            inside_ratio = frames_inside_zone / total_frames_seen
            if inside_ratio > self.threshold:
                staff_ids.append(track_id)

        return staff_ids
