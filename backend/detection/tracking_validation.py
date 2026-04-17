import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class TrackValidationStats:
    track_id: int
    first_frame: int
    last_frame: int
    detected_frames: int = 0
    not_detected_frames: int = 0
    expected_visible_frames: int = 0
    actual_visible_frames: int = 0
    mismatch_frames: int = 0
    max_consecutive_missing_frames: int = 0

    _current_missing_streak: int = 0

    def update_missing_streak(self, detected: bool) -> None:
        if detected:
            self._current_missing_streak = 0
            return
        self._current_missing_streak += 1
        if self._current_missing_streak > self.max_consecutive_missing_frames:
            self.max_consecutive_missing_frames = self._current_missing_streak


class TrackingValidationLogger:
    """
    Records an "expected vs actual" visibility audit for tracked boxes.

    Expected-visible definition (per track, per frame):
      - visible if detected in this frame, OR
      - visible if (now - last_seen) <= max_missing_seconds (grace period)

    Actual-visible definition (per track, per frame):
      - visible if the drawing code actually drew a box for this track_id

    Output:
      - CSV: per-frame rows (track_id, expected_visible, actual_visible, etc.)
      - JSON: end-of-run summary (per-id + overall)
    """

    def __init__(self, fps: int, csv_path: str, summary_json_path: str):
        self.fps = int(fps) if fps else 0
        self.csv_path = csv_path
        self.summary_json_path = summary_json_path

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(summary_json_path) or ".", exist_ok=True)

        self._fh = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow([
            "frame_idx",
            "track_id",
            "detected",
            "expected_visible",
            "actual_visible",
            "missing_for_s",
            "mismatch",
        ])

        self._stats: Dict[int, TrackValidationStats] = {}
        self._total_rows = 0

    def log_frame_for_track(
        self,
        frame_idx: int,
        track_id: int,
        detected: bool,
        expected_visible: bool,
        actual_visible: bool,
        missing_for_s: Optional[float],
    ) -> None:
        mismatch = expected_visible != actual_visible
        self._writer.writerow([
            frame_idx,
            track_id,
            1 if detected else 0,
            1 if expected_visible else 0,
            1 if actual_visible else 0,
            f"{missing_for_s:.3f}" if missing_for_s is not None else "",
            1 if mismatch else 0,
        ])
        self._total_rows += 1

        st = self._stats.get(track_id)
        if st is None:
            st = TrackValidationStats(track_id=track_id, first_frame=frame_idx, last_frame=frame_idx)
            self._stats[track_id] = st
        st.last_frame = frame_idx
        if detected:
            st.detected_frames += 1
        else:
            st.not_detected_frames += 1
        if expected_visible:
            st.expected_visible_frames += 1
        if actual_visible:
            st.actual_visible_frames += 1
        if mismatch:
            st.mismatch_frames += 1
        st.update_missing_streak(detected=detected)

    def close(self) -> None:
        try:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()
        finally:
            self._write_summary()

    def _write_summary(self) -> None:
        fps = self.fps if self.fps > 0 else None
        per_id = []
        total_mismatch = 0
        for st in sorted(self._stats.values(), key=lambda s: s.track_id):
            total_mismatch += st.mismatch_frames
            row = asdict(st)
            # internal helper field not useful for reports
            row.pop("_current_missing_streak", None)
            if fps:
                row["max_consecutive_missing_seconds"] = st.max_consecutive_missing_frames / fps
            per_id.append(row)

        summary = {
            "fps": fps,
            "total_tracks": len(self._stats),
            "total_rows": self._total_rows,
            "total_mismatch_frames": total_mismatch,
            "per_id": per_id,
        }

        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

