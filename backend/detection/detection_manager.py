import logging
import cv2
from app_paths import LOGS_DIR, ensure_data_dirs
from .video_processor import VideoProcessor
from .person_tracker import PersonTracker
from .statistics_logger import StatisticsLogger

class DetectionManager:
    def __init__(self, video_path, output_path, snaphots_path=None):
        self.processor = VideoProcessor(video_path, output_path)
        # Pass FPS to tracker so validation can convert frames->seconds accurately
        self.tracker = PersonTracker(
            snaphots_path=snaphots_path,
            video_fps=self.processor.fps,
        )

    def run(self):
        last_orig = None
        while True:
            frame = self.processor.read_frame()
            if frame is None:
                break
            orig_frame = frame.copy()
            last_orig = orig_frame
            processed_frame = self.tracker.process_frame(frame, orig_frame)
            self.processor.show_and_write(processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Tracks still visible on last frame (or never met dwell threshold) still need a snapshot folder.
        if last_orig is not None:
            self.tracker.finalize_pending_snapshots(last_orig)

        self.processor.release()
        staff_ids = self.tracker.staff_detector.get_staff_ids()
        StatisticsLogger.finalize(self.tracker.times, staff_ids=staff_ids)
        self.tracker.finalize_validation()


if __name__ == "__main__":
    ensure_data_dirs()
    logging.basicConfig(
        filename=str(LOGS_DIR / "people_tracking.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    app = DetectionManager("vid/coffee_shop.mp4", "processed_video.mp4", None)
    app.run()
