"""
HUD — overlay rendering for the drone camera frame.
"""
import cv2
import numpy as np

from simulator.physics.drone_state import DroneState
from simulator.terrain.orthophoto_map import OrthophotoMap


class HUD:
    """Renders Heads-Up Display over the camera frame."""

    def __init__(self, ortho_map: OrthophotoMap):
        self.ortho_map = ortho_map
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 1
        self.text_color = (0, 255, 0)      # Green text
        self.bg_color = (0, 0, 0)          # Black background
        self.bg_alpha = 0.5                # Semi-transparent background
        self._frame_count = 0
        self._last_gps = (0.0, 0.0)

    def render(
        self,
        frame: np.ndarray,
        state: DroneState,
        mode_name: str,
        fps: float,
        gsd: float,
        progress: str = "",
        P_matrix: np.ndarray | None = None,
        waypoints: list = None,
        current_wp_idx: int = 0
    ) -> np.ndarray:
        """
        Draw HUD onto the frame.

        Args:
            frame: Input BGR image from camera renderer.
            state: Current drone state.
            mode_name: "MANUAL" or "AUTO".
            fps: Current rendering FPS.
            gsd: Ground Sample Distance (m/px).
            progress: Waypoint progress string (for AUTO mode).

        Returns:
            Frame with HUD overlay.
        """
        # Convert local position to GPS (throttled to save pyproj overhead)
        if self._frame_count % 10 == 0:
            self._last_gps = self.ortho_map.local_to_gps(state.position[0], state.position[1])
        self._frame_count += 1
        lat, lon = self._last_gps

        # Prepare text lines
        lines = [
            f"MODE: {mode_name}",
            f"FPS:  {fps:.1f}",
            f"ALT:  {state.altitude:.1f} m",
            f"SPD:  {state.speed_3d:.1f} m/s",
            f"LAT:  {lat:.6f}",
            f"LON:  {lon:.6f}",
            f"GSD:  {gsd*100:.2f} cm/px",
        ]
        
        if progress:
            lines.append(f"WP:   {progress}")

        # Calculate background box size based on longest text
        max_len = max(len(line) for line in lines)
        box_w = int(max_len * 12 * self.font_scale)
        box_h = int(len(lines) * 25 * self.font_scale) + 10
        
        # Draw semi-transparent background for text (only on ROI, avoids full frame copy)
        roi = frame[10:10+box_h, 10:10+box_w]
        blk = np.empty_like(roi)
        blk[:] = self.bg_color
        cv2.addWeighted(roi, 1.0 - self.bg_alpha, blk, self.bg_alpha, 0, roi)

        # Draw text
        y = 30
        for line in lines:
            cv2.putText(frame, line, (20, y), self.font, self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)
            y += int(25 * self.font_scale)

        # Draw trajectory overlay
        if waypoints and P_matrix is not None:
            def draw_clipped_line(wp1, wp2, color, thickness):
                X1 = np.array([wp1.x, wp1.y, 0.0, 1.0])
                X2 = np.array([wp2.x, wp2.y, 0.0, 1.0])
                
                p1_cam = P_matrix @ X1
                p2_cam = P_matrix @ X2
                
                z1 = p1_cam[2]
                z2 = p2_cam[2]
                
                if z1 <= 0 and z2 <= 0:
                    return # Both behind camera
                
                if z1 <= 0:
                    t = (1e-2 - z1) / (z2 - z1)
                    p1_cam = p1_cam + t * (p2_cam - p1_cam)
                elif z2 <= 0:
                    t = (1e-2 - z2) / (z1 - z2)
                    p2_cam = p2_cam + t * (p1_cam - p2_cam)
                    
                u1, v1 = int(p1_cam[0] / p1_cam[2]), int(p1_cam[1] / p1_cam[2])
                u2, v2 = int(p2_cam[0] / p2_cam[2]), int(p2_cam[1] / p2_cam[2])
                
                # Clamp coordinates to avoid OpenCV integer overflow
                u1 = max(-30000, min(30000, u1))
                v1 = max(-30000, min(30000, v1))
                u2 = max(-30000, min(30000, u2))
                v2 = max(-30000, min(30000, v2))
                
                cv2.line(frame, (u1, v1), (u2, v2), color, thickness)

            # Draw line from drone to current waypoint
            if current_wp_idx < len(waypoints):
                wp_drone = type('Waypoint', (), {'x': state.position[0], 'y': state.position[1]})()
                draw_clipped_line(wp_drone, waypoints[current_wp_idx], (0, 165, 255), 2)  # Orange line for current transit

            # Draw lines between waypoints
            for i in range(len(waypoints) - 1):
                wp1 = waypoints[i]
                wp2 = waypoints[i+1]
                
                if i >= current_wp_idx:
                    color = (0, 255, 0) # Green for future path
                    thickness = 2
                elif i == current_wp_idx - 1:
                    color = (0, 200, 0) # Dark green for current segment
                    thickness = 2
                else:
                    color = (100, 100, 100) # Gray for past path
                    thickness = 1
                    
                draw_clipped_line(wp1, wp2, color, thickness)

        return frame
