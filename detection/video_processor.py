import cv2
import numpy as np
import pyautogui

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
