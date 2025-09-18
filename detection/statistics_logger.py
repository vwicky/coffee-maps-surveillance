import logging
import csv
from datetime import datetime

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
