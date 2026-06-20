"""
Telemetry Logger — saves drone state to a CSV file.
"""
import csv
import time
from pathlib import Path

from simulator.physics.drone_state import DroneState


class TelemetryLogger:
    """Logs drone telemetry to a CSV file periodically."""

    def __init__(self, output_file: str = "telemetry.csv", log_interval_frames: int = 15):
        """
        Args:
            output_file: Path to the output CSV file.
            log_interval_frames: Save telemetry every N frames.
        """
        self.output_file = Path(output_file)
        self.log_interval_frames = log_interval_frames
        self.frame_count = 0
        
        # Ensure parent directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Keep file open to avoid IO stutter
        self.file_handle = open(self.output_file, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file_handle)
        self.writer.writerow([
            "sys_time", "sim_time", 
            "pos_x", "pos_y", "alt_z", 
            "vel_x", "vel_y", "vel_z",
            "yaw_rad", "speed_3d"
        ])
            
    def log(self, state: DroneState) -> None:
        """Log state if the frame interval is reached."""
        self.frame_count += 1
        if self.frame_count % self.log_interval_frames != 0:
            return
            
        self.writer.writerow([
            f"{time.time():.3f}",
            f"{state.time:.3f}",
            f"{state.position[0]:.4f}", f"{state.position[1]:.4f}", f"{state.altitude:.4f}",
            f"{state.velocity[0]:.4f}", f"{state.velocity[1]:.4f}", f"{state.velocity[2]:.4f}",
            f"{state.yaw:.4f}",
            f"{state.speed_3d:.4f}"
        ])
        self.file_handle.flush()  # Ensure it writes to disk without closing

    def close(self) -> None:
        """Close the file handle safely."""
        if hasattr(self, 'file_handle') and not self.file_handle.closed:
            self.file_handle.flush()
            self.file_handle.close()
            
    def __del__(self) -> None:
        self.close()
