"""
Camera renderer — extracts a camera frame from the orthophoto based on drone position.

This replaces Unity rendering: given the drone's (x, y, z, yaw), it crops and rotates
the appropriate region of the orthophoto to produce a nadir camera view.

Rendering contract:
- CPU is the deterministic full-resolution reference path.
- GPU uses the same camera homography, DEM georeferencing, displacement formula,
  invalid-ray policy and bilinear interpolation.
- Frame caching when drone state hasn't changed
- Inline quaternion math (avoids scipy.Rotation overhead)
"""
import cv2
import numpy as np

from simulator.camera.camera_model import CameraModel
from simulator.terrain.parallax import apply_parallax, passes_for_altitude
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

    def __init__(
        self,
        camera: CameraModel,
        ortho_map: OrthophotoMap,
        renderer: str = "auto",
    ):
        """
        Args:
            camera: Camera model with resolution and lens parameters.
            ortho_map: Loaded orthophoto map.
        """
        self.camera = camera
        self.ortho_map = ortho_map
        if renderer not in {"auto", "cpu", "gpu"}:
            raise ValueError("renderer must be one of: auto, cpu, gpu")
        self.renderer_requested = renderer

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
        res_x = getattr(self.ortho_map, "local_res_x", self.ortho_map.res_x)
        res_y = getattr(self.ortho_map, "local_res_y", self.ortho_map.res_y)
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
            if renderer == "cpu":
                raise ImportError("CPU renderer explicitly requested")
            import cupy as cp
            self.use_gpu = True
            # Transfer image to GPU. Cast to float32 for interpolation.
            self.gpu_image = cp.asarray(self.ortho_map.image, dtype=cp.float32)
            if self.ortho_map.elevation is not None:
                self.gpu_elevation = cp.asarray(self.ortho_map.elevation, dtype=cp.float32)
                self.gpu_map_to_elevation = cp.asarray(
                    self.ortho_map.map_to_elevation_pixel_affine(), dtype=cp.float32
                )
            else:
                self.gpu_elevation = None
                self.gpu_map_to_elevation = None
                
            # Pre-initialize pixel grid
            V, U = cp.meshgrid(cp.arange(self.out_h), cp.arange(self.out_w), indexing='ij')
            self.gpu_grid = cp.stack([U.flatten(), V.flatten(), cp.ones_like(U).flatten()], axis=0).astype(cp.float32)
            
            # Pre-allocate output buffer on GPU
            self._gpu_frame_buf = cp.empty((self.out_h * self.out_w, 3), dtype=cp.uint8)
            
            print("[CameraRenderer] CuPy initialized. Rendering on GPU.")
        except ImportError:
            if renderer == "gpu":
                raise RuntimeError("GPU renderer requested but CuPy is unavailable")
            self.use_gpu = False
            self.gpu_image = None
            self.gpu_elevation = None
            self.gpu_map_to_elevation = None
            print("[CameraRenderer] CuPy not found. Falling back to CPU rendering.")
        except ValueError as exc:
            if renderer == "gpu":
                raise RuntimeError(str(exc)) from exc
            self.use_gpu = False
            self.gpu_image = None
            self.gpu_elevation = None
            self.gpu_map_to_elevation = None
            print(f"[CameraRenderer] GPU geometry unavailable ({exc}). Using CPU.")

        self.renderer_name = "gpu" if self.use_gpu else "cpu"
        self.last_surface_valid_fraction = 1.0

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
        """Compute the final homography for a given intrinsic matrix K.

        NOTE: this must NOT touch ``self.last_P``. The CPU path calls it twice —
        full-res then half-res — so assigning here left ``last_P`` holding the
        HALF-resolution projection matrix. CalibrationLogger builds anchors from
        ``last_P`` with FULL-resolution pixel coordinates, so every anchor
        recorded without CuPy came out at half scale. The caller assigns
        ``last_P`` explicitly from the full-res matrix.
        """
        # 3. Projection Matrix P = K [R | T]
        P = K @ np.hstack((R_world_to_cam, T_cam.reshape(3, 1)))
        
        # 4. Homography H1 mapping World Ground (X, Y, Z=0) to Camera Pixels
        # Ground points are [X, Y, 0, 1]^T, so we take cols 0, 1, 3 of P
        H1 = P[:, [0, 1, 3]]
        
        # 5. Final Homography: map_pixels -> camera_pixels
        return H1 @ self.H_map, P

    def _render_cpu(self, R_world_to_cam, T_cam, altitude):
        """Reference CPU ray-map renderer, including the same DEM model as GPU."""
        H_final_full, P_full = self._compute_homography(R_world_to_cam, T_cam, self.K)
        self.last_P = P_full
        try:
            H_inv = np.linalg.inv(H_final_full)
        except np.linalg.LinAlgError:
            self.last_surface_valid_fraction = 0.0
            return np.full((self.out_h, self.out_w, 3), 30, dtype=np.uint8)

        vv, uu = np.mgrid[0:self.out_h, 0:self.out_w]
        grid = np.stack((uu.ravel(), vv.ravel(), np.ones(uu.size)), axis=0)
        mapped = H_inv @ grid
        denominator = mapped[2]
        # H_inv returns a homogeneous ground point whose denominator has the
        # same sign as camera depth. Negative/zero depth is behind the camera
        # or on the horizon and must not be rendered as plausible terrain.
        valid = np.isfinite(denominator) & (denominator > 1e-9)
        u_map = np.full(denominator.shape, np.nan, dtype=np.float64)
        v_map = np.full(denominator.shape, np.nan, dtype=np.float64)
        u_map[valid] = mapped[0, valid] / denominator[valid]
        v_map[valid] = mapped[1, valid] / denominator[valid]

        if self.ortho_map.elevation is not None:
            lx = -float(T_cam @ R_world_to_cam[:, 0])
            ly = -float(T_cam @ R_world_to_cam[:, 1])
            drone_col, drone_row = self.ortho_map.local_to_pixel(lx, ly)

            def sample_height(u, v):
                return self.ortho_map.sample_elevation_at_map_pixels(u, v)

            u_map, v_map = apply_parallax(
                u_map,
                v_map,
                drone_col,
                drone_row,
                altitude,
                sample_height,
                self.ortho_map.base_elevation,
                num_passes=passes_for_altitude(altitude),
            )

        valid &= np.isfinite(u_map) & np.isfinite(v_map)
        valid &= (
            (u_map >= 0)
            & (u_map <= self.ortho_map.width - 1)
            & (v_map >= 0)
            & (v_map <= self.ortho_map.height - 1)
        )
        self.last_surface_valid_fraction = float(np.mean(valid))
        u_map = np.where(valid, u_map, -1).reshape(self.out_h, self.out_w).astype(np.float32)
        v_map = np.where(valid, v_map, -1).reshape(self.out_h, self.out_w).astype(np.float32)
        return cv2.remap(
            self.ortho_map.image,
            u_map,
            v_map,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(30, 30, 30),
        )

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
            self.last_surface_valid_fraction = 0.0
            return np.full((self.out_h, self.out_w, 3), 30, dtype=np.uint8)
        
        H_inv_cp = cp.asarray(H_inv, dtype=cp.float32)
        
        # Manual matrix multiplication to avoid cuBLAS DLL dependency on Windows
        u_map = H_inv_cp[0, 0] * self.gpu_grid[0] + H_inv_cp[0, 1] * self.gpu_grid[1] + H_inv_cp[0, 2] * self.gpu_grid[2]
        v_map = H_inv_cp[1, 0] * self.gpu_grid[0] + H_inv_cp[1, 1] * self.gpu_grid[1] + H_inv_cp[1, 2] * self.gpu_grid[2]
        w_map = H_inv_cp[2, 0] * self.gpu_grid[0] + H_inv_cp[2, 1] * self.gpu_grid[1] + H_inv_cp[2, 2] * self.gpu_grid[2]
        
        valid_ray = cp.isfinite(w_map) & (w_map > 1e-7)
        safe_w = cp.where(valid_ray, w_map, 1.0)
        u_map /= safe_w
        v_map /= safe_w
        
        # Parallax Displacement Mapping (3D Terrain) — adaptive passes.
        # The formula itself lives in simulator.terrain.parallax so that
        # CalibrationLogger displaces its five anchor points by EXACTLY the same
        # rule; anchors used to assume the flat Z=0 plane and silently disagreed
        # with the rendered image by r*h_rel/altitude (audit 2026-08-01).
        if self.gpu_elevation is not None:
            drone_col, drone_row = self.ortho_map.local_to_pixel(lx, ly)
            def _sample_h_abs(u, v):
                A = self.gpu_map_to_elevation
                col = A[0, 0] * u + A[0, 1] * v + A[0, 2]
                row = A[1, 0] * u + A[1, 1] * v + A[1, 2]
                coords = cp.stack([row, col], axis=0)
                return cupyx.scipy.ndimage.map_coordinates(
                    self.gpu_elevation, coords, order=1, mode='constant', cval=cp.nan
                )

            u_map, v_map = apply_parallax(
                u_map.copy(),
                v_map.copy(),
                drone_col,
                drone_row,
                altitude,
                _sample_h_abs,
                self.ortho_map.base_elevation,
                num_passes=passes_for_altitude(altitude),
            )

        valid = valid_ray & cp.isfinite(u_map) & cp.isfinite(v_map)
        valid &= (
            (u_map >= 0)
            & (u_map <= self.ortho_map.width - 1)
            & (v_map >= 0)
            & (v_map <= self.ortho_map.height - 1)
        )
        self.last_surface_valid_fraction = float(cp.mean(valid).get())
        u_map = cp.where(valid, u_map, -1)
        v_map = cp.where(valid, v_map, -1)
        
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
