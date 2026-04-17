import logging


class PhantomFilter:
    def __init__(self, frame_width: int, frame_height: int, buffer_percent: float = 0.15):
        """
        frame_width, frame_height: video resolution
        buffer_percent: width of the allowed edge-entry zone (0.15 = 15%)
        """
        self.w = frame_width
        self.h = frame_height
        self.buffer = buffer_percent

        # Per-track decision cache: once a track is validated, keep that result.
        self.id_status: dict[int, bool] = {}

    def check_and_filter(self, track_id: int, bbox: list[int] | tuple[int, int, int, int]) -> bool:
        """
        Return True when the track looks like a ghost/reflection and should be ignored.
        """
        # Keep the first decision for this track ID so a valid person can later walk
        # through the frame center without being filtered out.
        if track_id in self.id_status:
            return self.id_status[track_id]

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        margin_x = self.w * self.buffer
        margin_y = self.h * self.buffer

        is_in_center = (
            margin_x < center_x < (self.w - margin_x)
            and margin_y < center_y < (self.h - margin_y)
        )

        self.id_status[track_id] = is_in_center
        if is_in_center:
            logging.info(
                "PHANTOM_FILTER person=%s center=(%.1f,%.1f) bbox=(%s,%s,%s,%s)",
                track_id,
                center_x,
                center_y,
                x1,
                y1,
                x2,
                y2,
            )

        return is_in_center
