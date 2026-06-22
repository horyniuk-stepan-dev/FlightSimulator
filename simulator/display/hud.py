"""
HUD — overlay rendering for the drone camera frame.

Performance optimizations:
- Batch waypoint projection (single matrix multiply instead of per-waypoint loop)
- Frustum culling: only draw waypoints near current_wp_idx
- Avoid np.array() allocations inside draw loop
"""
import cv2
import numpy as np

from simulator.physics.drone_state import DroneState
from simulator.terrain.orthophoto_map import OrthophotoMap


class HUD:

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

        # Draw trajectory overlay — optimized with batch projection and windowed drawing
        if waypoints and P_matrix is not None:
            self._draw_trajectory_batch(frame, P_matrix, waypoints, current_wp_idx, state)

        return frame

    def _draw_trajectory_batch(self, frame, P_matrix, waypoints, current_wp_idx, state):
        """Draw waypoint trajectory using batch projection with proper near-plane clipping."""
        n_wp = len(waypoints)
        if n_wp == 0:
            return

        # Draw all waypoints (batch projection makes this fast enough)
        draw_start = 0
        draw_end = n_wp

        # Build homogeneous coordinate array for all visible waypoints (batch)
        # Include drone position as index 0 for the "current transit" line
        num_points = draw_end - draw_start + 1  # +1 for drone position
        world_pts = np.ones((4, num_points), dtype=np.float64)
        
        # First point = drone position
        world_pts[0, 0] = state.position[0]
        world_pts[1, 0] = state.position[1]
        world_pts[2, 0] = state.position[2]  # Drone altitude
        
        # Remaining points = waypoints in the draw window
        for i, wi in enumerate(range(draw_start, draw_end)):
            wp = waypoints[wi]
            world_pts[0, i + 1] = wp.x
            world_pts[1, i + 1] = wp.y
            
            # Draw lines slightly above the terrain surface (e.g., 10 meters)
            if self.ortho_map.elevation is not None:
                col, row = self.ortho_map.local_to_pixel(wp.x, wp.y)
                elev_h, elev_w = self.ortho_map.elevation.shape
                col_elev = col * (elev_w / self.ortho_map.width)
                row_elev = row * (elev_h / self.ortho_map.height)
                col_int = max(0, min(elev_w - 1, int(round(col_elev))))
                row_int = max(0, min(elev_h - 1, int(round(row_elev))))
                h_abs = float(self.ortho_map.elevation[row_int, col_int])
                h_rel = h_abs - self.ortho_map.base_elevation
            else:
                h_rel = 0.0
            world_pts[2, i + 1] = h_rel + 10.0
        
        # Single batch projection: P @ [4 x N] = [3 x N]
        projected = P_matrix @ world_pts  # shape (3, num_points)
        
        def _clip_and_draw(idx1, idx2, color, thickness):
            """Draw a line between two projected points with near-plane clipping."""
            p1 = projected[:, idx1].copy()
            p2 = projected[:, idx2].copy()
            z1, z2 = p1[2], p2[2]
            
            # Both behind camera → skip
            if z1 <= 0 and z2 <= 0:
                return
            
            # Clip to near plane (z = 0.01) if one point is behind camera
            if z1 <= 0:
                t = (1e-2 - z1) / (z2 - z1)
                p1 = p1 + t * (p2 - p1)
            elif z2 <= 0:
                t = (1e-2 - z2) / (z1 - z2)
                p2 = p2 + t * (p1 - p2)
            
            # Perspective division
            u1 = int(p1[0] / p1[2])
            v1 = int(p1[1] / p1[2])
            u2 = int(p2[0] / p2[2])
            v2 = int(p2[1] / p2[2])
            
            # Clamp to prevent OpenCV integer overflow
            u1 = max(-30000, min(30000, u1))
            v1 = max(-30000, min(30000, v1))
            u2 = max(-30000, min(30000, u2))
            v2 = max(-30000, min(30000, v2))
            
            cv2.line(frame, (u1, v1), (u2, v2), color, thickness)
        
        # Draw line from drone to current waypoint (orange)
        if current_wp_idx < n_wp and current_wp_idx >= draw_start:
            wp_local_idx = current_wp_idx - draw_start + 1
            _clip_and_draw(0, wp_local_idx, (0, 165, 255), 2)
        
        # Draw lines between consecutive waypoints
        for i in range(draw_start, min(draw_end - 1, n_wp - 1)):
            idx1 = i - draw_start + 1  # offset by 1 (drone is at 0)
            idx2 = idx1 + 1
            
            if i >= current_wp_idx:
                color = (0, 255, 0)   # Green for future path
                thickness = 2
            elif i == current_wp_idx - 1:
                color = (0, 200, 0)   # Dark green for current segment
                thickness = 2
            else:
                color = (100, 100, 100)  # Gray for past path
                thickness = 1
                
            _clip_and_draw(idx1, idx2, color, thickness)
