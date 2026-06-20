"""
Camera renderer — extracts a camera frame from the orthophoto based on drone position.

This replaces Unity rendering: given the drone's (x, y, z, yaw), it crops and rotates
the appropriate region of the orthophoto to produce a nadir camera view.
"""
import cv2
import numpy as np

from simulator.camera.camera_model import CameraModel
from simulator.terrain.orthophoto_map import OrthophotoMap
from simulator.physics.drone_state import DroneState


class CameraRenderer:
    """Renders nadir camera frames from the orthophoto based on drone state."""

    def __init__(self, camera: CameraModel, ortho_map: OrthophotoMap):
        """
        Args:
            camera: Camera model with resolution and lens parameters.
            ortho_map: Loaded orthophoto map.
        """
        self.camera = camera
        self.ortho_map = ortho_map

        # Load the orthophoto image into GPU memory using CuPy if available
        try:
            import cupy as cp
            self.use_gpu = True
            # Transfer image to GPU. Cast to float32 for interpolation.
            self.gpu_image = cp.asarray(self.ortho_map.image, dtype=cp.float32)
            print("[CameraRenderer] CuPy initialized. Rendering on GPU.")
        except ImportError:
            self.use_gpu = False
            self.gpu_image = None
            print("[CameraRenderer] CuPy not found. Falling back to CPU rendering.")

    def render(self, state: DroneState) -> np.ndarray:
        """
        Render a camera frame for the current drone state.

        Args:
            state: Current drone state.

        Returns:
            BGR image of shape (image_height_px, image_width_px, 3).
        """
        # Get drone position and orientation
        lx, ly = state.position[0], state.position[1]
        altitude = max(state.altitude, 1.0)
        
        out_w = self.camera.image_width_px
        out_h = self.camera.image_height_px
        
        # 1. Camera Intrinsics K
        f_px = self.camera.focal_length_mm * (out_w / self.camera.sensor_width_mm)
        cx = out_w / 2.0
        cy = out_h / 2.0
        
        K = np.array([
            [f_px, 0.0, cx],
            [0.0, f_px, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # 2. Camera Extrinsics
        from scipy.spatial.transform import Rotation
        R_body_to_world = Rotation.from_quat(state.quaternion).as_matrix()
        R_world_to_body = R_body_to_world.T
        
        # Fixed mount: Camera X=Right, Y=Down, Z=Forward | Drone X=Right, Y=Forward, Z=Up
        R_body_to_cam = np.array([
            [ 1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1]
        ], dtype=np.float64)
        
        R_world_to_cam = R_body_to_cam @ R_world_to_body
        
        C_world = np.array([lx, ly, altitude], dtype=np.float64)
        T_cam = -R_world_to_cam @ C_world
        
        # 3. Projection Matrix P = K [R | T]
        P = K @ np.hstack((R_world_to_cam, T_cam.reshape(3, 1)))
        self.last_P = P
        
        # 4. Homography H1 mapping World Ground (X, Y, Z=0) to Camera Pixels
        # Ground points are [X, Y, 0, 1]^T, so we take cols 0, 1, 3 of P
        H1 = P[:, [0, 1, 3]]
        
        # 5. Homography H_map mapping Orthophoto Pixels to World Ground
        res_x = self.ortho_map.res_x
        res_y = self.ortho_map.res_y
        map_w = self.ortho_map.width
        map_h = self.ortho_map.height
        
        H_map = np.array([
            [res_x, 0,      -(map_w / 2.0) * res_x],
            [0,     -res_y,  (map_h / 2.0) * res_y],
            [0,     0,      1.0]
        ], dtype=np.float64)
        
        # 6. Final Homography: map_pixels -> camera_pixels
        H_final = H1 @ H_map
        
        # We need H_inv to map camera_pixels -> map_pixels for rendering
        try:
            H_inv = np.linalg.inv(H_final)
        except np.linalg.LinAlgError:
            # Fallback if singular (e.g. looking exactly parallel to ground from 0 altitude)
            H_inv = np.eye(3, dtype=np.float64)

        if self.use_gpu:
            import cupy as cp
            import cupyx.scipy.ndimage
            
            # Lazy initialize the pixel grid to save time
            if not hasattr(self, 'gpu_grid'):
                V, U = cp.meshgrid(cp.arange(out_h), cp.arange(out_w), indexing='ij')
                self.gpu_grid = cp.stack([U.flatten(), V.flatten(), cp.ones_like(U).flatten()], axis=0).astype(cp.float32)
                
            H_inv_cp = cp.asarray(H_inv, dtype=cp.float32)
            
            # Manual matrix multiplication to avoid cuBLAS DLL dependency on Windows
            u_map = H_inv_cp[0, 0] * self.gpu_grid[0] + H_inv_cp[0, 1] * self.gpu_grid[1] + H_inv_cp[0, 2] * self.gpu_grid[2]
            v_map = H_inv_cp[1, 0] * self.gpu_grid[0] + H_inv_cp[1, 1] * self.gpu_grid[1] + H_inv_cp[1, 2] * self.gpu_grid[2]
            w_map = H_inv_cp[2, 0] * self.gpu_grid[0] + H_inv_cp[2, 1] * self.gpu_grid[1] + H_inv_cp[2, 2] * self.gpu_grid[2]
            
            # Perspective division (add small epsilon to avoid div by zero)
            w_div = w_map + 1e-7
            u_map /= w_div
            v_map /= w_div
            
            coords = cp.stack([v_map, u_map], axis=0)
            
            frame_gpu = cp.empty((out_h * out_w, 3), dtype=cp.uint8)
            for c in range(3):
                channel = cupyx.scipy.ndimage.map_coordinates(
                    self.gpu_image[:, :, c],
                    coords,
                    order=3,  # Bicubic interpolation for better quality
                    mode='constant',
                    cval=30.0
                )
                frame_gpu[:, c] = cp.clip(channel, 0, 255).astype(cp.uint8)
                
            frame = cp.asnumpy(frame_gpu.reshape((out_h, out_w, 3)))
            return frame
        else:
            # OpenCV warping (CPU)
            # warpPerspective applies M to output pixel coords to find input coords IF WARP_INVERSE_MAP is used.
            # So if we use WARP_INVERSE_MAP, M should be H_final (src->dst).
            frame = cv2.warpPerspective(
                self.ortho_map.image,
                H_final,
                (out_w, out_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(30, 30, 30),
            )
            return frame
