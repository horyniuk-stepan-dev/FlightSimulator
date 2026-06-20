"""Quick diagnostic test for the simulator pipeline."""
import sys
import time
import traceback
import numpy as np

print("=== Simulator Diagnostic ===")

# 1. Test terrain loading
print("\n[1] Testing terrain loading...")
try:
    from simulator.terrain.orthophoto_map import OrthophotoMap
    ortho = OrthophotoMap(".tile_cache/terrain_d3bd4c2d659e_z19.tif")
    print(f"    Image shape: {ortho.image.shape}, dtype: {ortho.image.dtype}")
    print(f"    Image min/max: {ortho.image.min()} / {ortho.image.max()}")
    print(f"    Bounds (local): {ortho.get_bounds_local()}")
    print(f"    Center pixel test: {ortho.local_to_pixel(0, 0)}")
    print("    [OK] Terrain loaded successfully")
except Exception as e:
    print(f"    [FAIL] {e}")
    traceback.print_exc()

# 2. Test camera model
print("\n[2] Testing camera model...")
try:
    from simulator.camera.camera_model import CameraModel
    from simulator.config import CameraConfig
    cam = CameraModel.from_config(CameraConfig())
    print(f"    FOV horizontal: {np.degrees(cam.fov_horizontal_rad):.1f} deg")
    print(f"    Footprint at 100m: {cam.footprint_meters(100)} m")
    print(f"    GSD at 100m: {cam.gsd_m_per_px(100)*100:.2f} cm/px")
    print("    [OK] Camera model works")
except Exception as e:
    print(f"    [FAIL] {e}")
    traceback.print_exc()

# 3. Test camera renderer
print("\n[3] Testing camera renderer...")
try:
    from simulator.camera.camera_renderer import CameraRenderer
    from simulator.physics.drone_state import DroneState
    renderer = CameraRenderer(cam, ortho)
    state = DroneState(position=np.array([0.0, 0.0, 100.0]))
    frame = renderer.render(state)
    print(f"    Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"    Frame min/max: {frame.min()} / {frame.max()}")
    print(f"    Non-zero pixels: {np.count_nonzero(frame)} / {frame.size}")
    print("    [OK] Camera renderer works")
except Exception as e:
    print(f"    [FAIL] {e}")
    traceback.print_exc()

# 4. Test RotorPy flight controller
print("\n[4] Testing RotorPy flight controller...")
try:
    from simulator.physics.flight_controller import FlightController
    from simulator.config import PhysicsConfig
    fc = FlightController(PhysicsConfig(), initial_position=np.array([0.0, 0.0, 100.0]))
    
    state0 = fc.get_state()
    print(f"    Initial pos: {state0.position}")
    print(f"    Initial vel: {state0.velocity}")
    print(f"    Initial alt: {state0.altitude}")
    
    # Step with zero velocity command (should hover)
    for i in range(10):
        state = fc.step(np.array([0.0, 0.0, 0.0]), 0.0, 0.01)
    print(f"    After 10 hover steps: pos={state.position}, vel={state.velocity}")
    print(f"    Altitude drift: {state.altitude - 100.0:.4f} m")
    
    # Step with forward velocity
    for i in range(10):
        state = fc.step(np.array([2.0, 0.0, 0.0]), 0.0, 0.01)
    print(f"    After 10 forward steps: pos={state.position}, vel={state.velocity}")
    print("    [OK] Flight controller works")
except Exception as e:
    print(f"    [FAIL] {e}")
    traceback.print_exc()

# 5. Test OpenCV display
print("\n[5] Testing OpenCV display...")
try:
    import cv2
    # Render a frame with HUD
    from simulator.display.hud import HUD
    hud = HUD(ortho)
    
    state = DroneState(position=np.array([0.0, 0.0, 100.0]))
    frame = renderer.render(state)
    hud_frame = hud.render(frame, state, mode_name="MANUAL", fps=30.0, gsd=cam.gsd_m_per_px(100.0))
    
    print(f"    HUD frame shape: {hud_frame.shape}")
    
    # Save test frame to disk
    cv2.imwrite("test_frame.png", hud_frame)
    print("    [OK] Test frame saved as test_frame.png")
    
    # Try displaying for 2 seconds
    cv2.namedWindow("Diagnostic Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Diagnostic Test", hud_frame)
    print("    Showing test frame for 3 seconds... (press any key to skip)")
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    print("    [OK] OpenCV display works")
except Exception as e:
    print(f"    [FAIL] {e}")
    traceback.print_exc()

print("\n=== Diagnostic Complete ===")
