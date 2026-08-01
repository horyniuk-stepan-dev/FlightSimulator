"""
FrameSink interface and display implementation.
"""
from abc import ABC, abstractmethod

import cv2
import numpy as np


class FrameSink(ABC):
    """Abstract base class for consuming rendered frames."""
    
    @abstractmethod
    def consume(self, frame: np.ndarray) -> None:
        """Process a single frame."""
        pass
        
    @abstractmethod
    def is_closed(self) -> bool:
        """Return True if the sink was closed/stopped by the user."""
        pass


class DisplaySink(FrameSink):
    """Displays frames using cv2.imshow."""
    
    def __init__(self, window_name: str = "Drone Simulator"):
        self.window_name = window_name
        self._closed = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
    def consume(self, frame: np.ndarray) -> None:
        if self._closed:
            return
        try:
            cv2.imshow(self.window_name, frame)
        except Exception:
            self._closed = True
        
    def cleanup(self) -> None:
        self._closed = True
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def is_closed(self) -> bool:
        if self._closed:
            return True
        try:
            vis = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            auto = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE)
            if vis < 1 or auto < 0:
                self._closed = True
                return True
        except Exception:
            self._closed = True
            return True
        return False
