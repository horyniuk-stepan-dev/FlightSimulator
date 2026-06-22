"""
Camera renderer — extracts a camera frame from the orthophoto based on drone position.

This replaces Unity rendering: given the drone's (x, y, z, yaw), it crops and rotates
the appropriate region of the orthophoto to produce a nadir camera view.

Performance optimizations:
- CPU: renders at half resolution then upscales (4x faster warpPerspective)
- CPU: adaptive interpolation quality (NEAREST at high altitude, LINEAR at low)
- GPU: bilinear interpolation (order=1) instead of bicubic (order=3)
- GPU: adaptive parallax passes (1 at altitude>200m, 3 below)
- Frame caching when drone state hasn't changed
- Inline quaternion math (avoids scipy.Rotation overhead)
"""
import cv2
import numpy as np

from simulator.camera.camera_model import CameraModel
from simulator.terrain.orthophoto_map import OrthophotoMap
from simulator.physics.drone_state import DroneState


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    
    Inline formula avoids scipy.spatial.transform.Rotation overhead (~0.5ms/call).
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    x2 = x + x
    y2 = y + y
    z2 = z + z
    xx = x * x2
    xy = x * y2
    xz = x * z2
    yy = y * y2
    yz = y * z2
    zz = z * z2
    wx = w * x2
    wy = w * y2
    wz = w * z2
    
    return np.array([
        [1.0 - (yy + zz), xy - wz,          xz + wy],
        [xy + wz,          1.0 - (xx + zz),  yz - wx],
        [xz - wy,          yz + wx,           1.0 - (xx + yy)]
    ], dtype=np.float64)


class CameraRenderer:
    """Renders nadir camera frames from the orthophoto based on drone state."""

    # Threshold for frame caching: skip re-render if drone barely moved
    _CACHE_POS_THRESHOLD = 0.01    # meters
    _CACHE_QUAT_THRESHOLD = 1e-5   # quaternion element delta

    def __init__(self, camera: CameraModel, ortho_map: OrthophotoMap):
        """
        Args:
            camera: Camera model with resolution and lens parameters.
            ortho_map: Loaded orthophoto map.
        """
        self.camera = camera
        self.ortho_map = ortho_map

        # Precompute static camera matrices
        self.out_w = self.camera.image_width_px
        self.out_h = self.camera.image_height_px
        
        # Half-resolution dimensions for CPU fast path
        self.half_w = self.out_w // 2
        self.half_h = self.out_h // 2
        
        f_px = self.camera.focal_length_mm * (self.out_w / self.camera.sensor_width_mm)
        cx = self.out_w / 2.0
        cy = self.out_h / 2.0
        
        self.K = np.array([
            [f_px, 0.0, cx],
            [0.0, f_px, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Half-resolution intrinsics for CPU fast path
        f_px_half = f_px * 0.5
        self.K_half = np.array([
            [f_px_half, 0.0,       cx * 0.5],
            [0.0,       f_px_half, cy * 0.5],
            [0.0,       0.0,       1.0]
        ], dtype=np.float64)
        
        # Fixed mount: Camera X=Right, Y=Down, Z=Forward | Drone X=Right, Y=Forward, Z=Up
        self.R_body_to_cam = np.array([
            [ 1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1]
        ], dtype=np.float64)

        # Precompute map homography (Orthophoto Pixels to World Ground)
        res_x = self.ortho_map.res_x
        res_y = self.ortho_map.res_y
        map_w = self.ortho_map.width
        map_h = self.ortho_map.height
        
        self.H_map = np.array([
            [res_x, 0,      -(map_w / 2.0) * res_x],
            [0,     -res_y,  (map_h / 2.0) * res_y],
            [0,     0,      1.0]
        ], dtype=np.float64)

        # Frame cache state
        self._cached_frame = None
        self._cached_pos = None
        self._cached_quat = None

        # Load the orthophoto image into GPU memory using CuPy if available
        try:
            import cupy as cp
            self.use_gpu = True
            # Transfer image to GPU. Cast to float32 for interpolation.
            self.gpu_image = cp.asarray(self.ortho_map.image, dtype=cp.float32)
            if self.ortho_map.elevation is not None:
                self.gpu_elevation = cp.asarray(self.ortho_map.elevation, dtype=cp.float32)
            else:
                self.gpu_elevation = None
                
            # Pre-initialize pixel grid
            V, U = cp.meshgrid(cp.arange(self.out_h), cp.arange(self.out_w), indexing='ij')
            self.gpu_grid = cp.stack([U.flatten(), V.flatten(), cp.ones_like(U).flatten()], axis=0).astype(cp.float32)
            
            # Pre-allocate output buffer on GPU
            self._gpu_frame_buf = cp.empty((self.out_h * self.out_w, 3), dtype=cp.uint8)
            
            print("[CameraRenderer] CuPy initialized. Rendering on GPU.")
        except ImportError:
            self.use_gpu = False
            self.gpu_image = None
            self.gpu_elevation = None
            print("[CameraRenderer] CuPy not found. Falling back to CPU rendering.")

    def _is_state_cached(self, state: DroneState) -> bool:
        """Check if the drone state is close enough to use cached frame."""
        if self._cached_frame is None or self._cached_pos is None:
            return False
        pos_delta = np.abs(state.position - self._cached_pos).max()
        quat_delta = np.abs(state.quaternion - self._cached_quat).max()
        return (pos_delta < self._CACHE_POS_THRESHOLD and 
                quat_delta < self._CACHE_QUAT_THRESHOLD)

    def render(self, state: DroneState) -> np.ndarray:
        """
        Render a camera frame for the current drone state.

        Args:
            state: Current drone state.

        Returns:
            BGR image of shape (image_height_px, image_width_px, 3).
        """
        # Frame cache: skip re-render if drone hasn't moved
        if self._is_state_cached(state):
            return self._cached_frame
        
        # Get drone position and orientation
        lx, ly = state.position[0], state.position[1]
        altitude = max(state.altitude, 1.0)
        
        # 2. Camera Extrinsics — inline quaternion→matrix (avoids scipy overhead)
        R_body_to_world = _quat_to_matrix(state.quaternion)
        R_world_to_body = R_body_to_world.T
        
        R_world_to_cam = self.R_body_to_cam @ R_world_to_body
        
        C_world = np.array([lx, ly, altitude], dtype=np.float64)
        T_cam = -R_world_to_cam @ C_world
        
        if self.use_gpu:
            frame = self._render_gpu(R_world_to_cam, T_cam, lx, ly, altitude)
        else:
            frame = self._render_cpu(R_world_to_cam, T_cam, altitude)
        
        # Update cache
        self._cached_frame = frame
        self._cached_pos = state.position.copy()
        self._cached_quat = state.quaternion.copy()
        
        return frame

    def _compute_homography(self, R_world_to_cam, T_cam, K):
        """Compute the final homography for a given intrinsic matrix K."""
        # 3. Projection Matrix P = K [R | T]
        P = K @ np.hstack((R_world_to_cam, T_cam.reshape(3, 1)))
        self.last_P = P
        
        # 4. Homography H1 mapping World Ground (X, Y, Z=0) to Camera Pixels
        # Ground points are [X, Y, 0, 1]^T, so we take cols 0, 1, 3 of P
        H1 = P[:, [0, 1, 3]]
        
        # 5. Final Homography: map_pixels -> camera_pixels
        return H1 @ self.H_map, P

    def _render_cpu(self, R_world_to_cam, T_cam, altitude):
        """CPU rendering path with half-resolution optimization."""
        # Always compute full-res P for HUD waypoint projection
        H_final_full, P_full = self._compute_homography(R_world_to_cam, T_cam, self.K)
        self.last_P = P_full
        
        # Render at half resolution for speed, then upscale
        H_final_half, _ = self._compute_homography(R_world_to_cam, T_cam, self.K_half)
        
        # Adaptive interpolation: NEAREST is faster at high altitudes where detail doesn't matter
        if altitude > 500.0:
            interp_flag = cv2.INTER_NEAREST
        else:
            interp_flag = cv2.INTER_LINEAR
        
        # warpPerspective at half resolution (4x fewer pixels = ~4x faster)
        frame_half = cv2.warpPerspective(
            self.ortho_map.image,
            H_final_half,
            (self.half_w, self.half_h),
            flags=interp_flag | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(30, 30, 30),
        )
        
        # Upscale to output resolution
        frame = cv2.resize(frame_half, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        return frame

    def _render_gpu(self, R_world_to_cam, T_cam, lx, ly, altitude):
        """GPU rendering path with optimized interpolation and adaptive parallax."""
        import cupy as cp
        import cupyx.scipy.ndimage
        
        H_final, P = self._compute_homography(R_world_to_cam, T_cam, self.K)
        self.last_P = P
        
        # We need H_inv to map camera_pixels -> map_pixels for rendering
        try:
            H_inv = np.linalg.inv(H_final)
        except np.linalg.LinAlgError:
            H_inv = np.eye(3, dtype=np.float64)
        
        H_inv_cp = cp.asarray(H_inv, dtype=cp.float32)
        
        # Manual matrix multiplication to avoid cuBLAS DLL dependency on Windows
        u_map = H_inv_cp[0, 0] * self.gpu_grid[0] + H_inv_cp[0, 1] * self.gpu_grid[1] + H_inv_cp[0, 2] * self.gpu_grid[2]
        v_map = H_inv_cp[1, 0] * self.gpu_grid[0] + H_inv_cp[1, 1] * self.gpu_grid[1] + H_inv_cp[1, 2] * self.gpu_grid[2]
        w_map = H_inv_cp[2, 0] * self.gpu_grid[0] + H_inv_cp[2, 1] * self.gpu_grid[1] + H_inv_cp[2, 2] * self.gpu_grid[2]
        
        # Perspective division (add small epsilon to avoid div by zero)
        w_div = w_map + 1e-7
        u_map /= w_div
        v_map /= w_div
        
        # Parallax Displacement Mapping (3D Terrain) — adaptive passes
        if self.gpu_elevation is not None:
            drone_col, drone_row = self.ortho_map.local_to_pixel(lx, ly)
            
            u_0 = u_map.copy()
            v_0 = v_map.copy()
            
            # Adaptive: fewer passes at high altitude where parallax effect is minimal
            num_passes = 1 if altitude > 200.0 else 3
            
            for _ in range(num_passes):
                elev_h, elev_w = self.gpu_elevation.shape
                coords = cp.stack([
                    v_map * (elev_h / self.ortho_map.height),
                    u_map * (elev_w / self.ortho_map.width)
                ], axis=0)
                h_abs = cupyx.scipy.ndimage.map_coordinates(
                    self.gpu_elevation, coords, order=1, mode='nearest'
                )
                # Height above the Z=0 base plane
                h_rel = h_abs - self.ortho_map.base_elevation
                
                # Avoid division by zero
                frac = h_rel / max(altitude, 10.0)
                
                # Shift coordinates towards the drone
                u_map = u_0 - frac * (u_0 - drone_col)
                v_map = v_0 - frac * (v_0 - drone_row)
        
        coords = cp.stack([v_map, u_map], axis=0)
        
        # Bilinear interpolation (order=1) instead of bicubic (order=3) — 2-3x faster
        for c in range(3):
            channel = cupyx.scipy.ndimage.map_coordinates(
                self.gpu_image[:, :, c],
                coords,
                order=1,  # Bilinear — much faster than bicubic, minimal quality loss
                mode='constant',
                cval=30.0
            )
            self._gpu_frame_buf[:, c] = cp.clip(channel, 0, 255).astype(cp.uint8)
            
        frame = cp.asnumpy(self._gpu_frame_buf.reshape((self.out_h, self.out_w, 3)))
        return frame
