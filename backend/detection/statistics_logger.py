import logging
import csv
from datetime import datetime
from pathlib import Path

from app_paths import LOGS_DIR, ensure_data_dirs

class StatisticsLogger:
    @staticmethod
    def finalize(times, staff_ids=None, csv_path: str | None = None):
        now = datetime.now()
        finalized_rows = []
        staff_ids = set(staff_ids or [])

        for tid, t in times.items():
            # Support both tracker schemas:
            # - {"enter", "last_seen"} from detection/person_tracker.py
            # - {"enter", "last", "exit", "duration"} from older tracker variants
            enter_time = t.get("enter")
            if enter_time is None:
                continue

            exit_time = t.get("exit") or t.get("last_seen") or t.get("last") or now
            duration = t.get("duration")
            if duration is None:
                duration = (exit_time - enter_time).total_seconds()

            t["exit"] = exit_time
            t["duration"] = duration
            first_bbox = t.get("first_bbox")
            last_bbox = t.get("last_bbox")
            max_area = t.get("max_area")
            frames_seen = t.get("frames_seen")
            max_conf = t.get("max_conf")
            snapshots_saved = 1 if isinstance(t.get("best_snapshot"), str) else 0
            is_person_staff = tid in staff_ids

            logging.info(
                "FINAL person=%s duration=%.1fs enter=%s exit=%s first_bbox=%s last_bbox=%s max_area=%s frames=%s max_conf=%s snapshots=%s is_staff=%s",
                tid,
                duration,
                enter_time.strftime("%H:%M:%S"),
                exit_time.strftime("%H:%M:%S"),
                first_bbox,
                last_bbox,
                max_area,
                frames_seen,
                f"{max_conf:.3f}" if max_conf is not None else None,
                snapshots_saved,
                is_person_staff,
            )

            finalized_rows.append({
                "id": tid,
                "enter": enter_time,
                "exit": exit_time,
                "duration": duration,
                "first_bbox": first_bbox,
                "last_bbox": last_bbox,
                "max_area": max_area,
                "frames_seen": frames_seen,
                "max_conf": max_conf,
                "snapshots_saved": snapshots_saved,
                "is_staff": is_person_staff,
            })

        total_visitors = len(times)
        if total_visitors > 0:
            durations = [r["duration"] for r in finalized_rows if r.get("duration") is not None]
            logging.info(f"Total visitors detected: {total_visitors}")
            if durations:
                logging.info(f"Average stay duration: {sum(durations)/len(durations):.1f}s")
                logging.info(f"Max stay duration: {max(durations):.1f}s")
                logging.info(f"Min stay duration: {min(durations):.1f}s")
                long_stay = sorted(finalized_rows, key=lambda r: r["duration"], reverse=True)[:3]
                logging.info(
                    "Top longest stays: %s",
                    ", ".join([f"id={r['id']}({r['duration']:.1f}s)" for r in long_stay])
                )
        else:
            logging.info("No visitors detected")

        # Save CSV
        ensure_data_dirs()
        output_path = Path(csv_path) if csv_path else (LOGS_DIR / "people_times.csv")
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
            for row in sorted(finalized_rows, key=lambda r: r["id"]):
                writer.writerow([
                    row["id"],
                    row["enter"].strftime("%Y-%m-%d %H:%M:%S") if row.get("enter") else "",
                    row["exit"].strftime("%Y-%m-%d %H:%M:%S") if row.get("exit") else "",
                    f"{row['duration']:.1f}" if row.get("duration") is not None else "",
                    row.get("frames_seen", ""),
                    f"{row['max_conf']:.3f}" if row.get("max_conf") is not None else "",
                    str(row.get("first_bbox", "")),
                    str(row.get("last_bbox", "")),
                    row.get("max_area", ""),
                    row.get("snapshots_saved", 0),
                    row.get("is_staff", False),
                ])
