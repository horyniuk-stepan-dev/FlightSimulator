"""
Video sink for recording frames to a video file.
"""
import cv2
import numpy as np

from simulator.display.frame_sink import FrameSink


class VideoWriterSink(FrameSink):
    """Writes frames to a video file."""

    def __init__(self, filename: str, fps: float, width: int, height: int):
        self.filename = filename
        self.fps = fps
        self.width = width
        self.height = height
        
        # Use mp4v codec for mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (self.width, self.height))
        
        if not self.writer.isOpened():
            print(f"[VideoWriterSink] Failed to open video writer for {self.filename}")
        else:
            print(f"[VideoWriterSink] Recording video to: {self.filename} ({self.width}x{self.height} @ {self.fps}fps)")

    def consume(self, frame: np.ndarray) -> None:
        """Write a single frame to the video."""
        if self.writer.isOpened():
            # If the frame size doesn't match, we need to resize it to prevent corruption
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))
            self.writer.write(frame)

    def is_closed(self) -> bool:
        return not self.writer.isOpened()

    def cleanup(self) -> None:
        """Release the video writer."""
        if self.writer.isOpened():
            self.writer.release()
            print(f"[VideoWriterSink] Video saved to {self.filename}")
