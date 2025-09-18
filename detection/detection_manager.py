import logging
import cv2
from video_processor import VideoProcessor
from person_tracker import PersonTracker
from statistics_logger import StatisticsLogger

class DetectionManager:
    def __init__(self, video_path, output_path, snaphots_path=None):
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
    logging.basicConfig(
        filename="logs/people_tracking.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    app = DetectionManager("vid/coffee_shop.mp4", "processed_video.mp4", None)
    app.run()
