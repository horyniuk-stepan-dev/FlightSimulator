"""
Satellite tile loader — downloads XYZ tiles for a bounding box and saves as GeoTIFF.
Uses contextily with Esri.WorldImagery (free, high-res up to zoom ~19).
"""
import os
import hashlib
from pathlib import Path
import threading
import time
import sys

import contextily as cx
import numpy as np


class ProgressTracker:
    def __init__(self, total_tiles):
        self.total_tiles = total_tiles
        self.downloaded = 0
        self.running = False
        self.thread = None

    def _spin(self):
        chars = "|/-\\"
        idx = 0
        start_time = time.time()
        while self.running:
            elapsed = time.time() - start_time
            if self.total_tiles > 0:
                percent = (self.downloaded / self.total_tiles) * 100
                sys.stdout.write(f"\r[TileLoader] Downloading tiles: {self.downloaded}/{self.total_tiles} ({percent:.1f}%) {chars[idx % len(chars)]} [{elapsed:.1f}s]")
            else:
                sys.stdout.write(f"\r[TileLoader] Downloading tiles: {self.downloaded} {chars[idx % len(chars)]} [{elapsed:.1f}s]")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)
        sys.stdout.write(f"\r{' ' * 80}\r")
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()


def bbox_hash(west: float, south: float, east: float, north: float, zoom: int) -> str:
    """Generate a short hash for caching based on bounding box and zoom."""
    key = f"{west:.6f}_{south:.6f}_{east:.6f}_{north:.6f}_z{zoom}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def download_tiles(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    zoom: int = 19,
    cache_dir: str = ".tile_cache",
    provider=None,
) -> str:
    """
    Download satellite tiles for the given bounding box and return path to GeoTIFF.

    Coordinates are in WGS84 (lat/lon).
    The resulting GeoTIFF is in Web Mercator (EPSG:3857).

    Args:
        lat_min, lon_min, lat_max, lon_max: Bounding box in WGS84.
        zoom: Tile zoom level (18-19 typically free).
        cache_dir: Directory for caching downloaded tiles.
        provider: contextily tile provider (default: Esri.WorldImagery).

    Returns:
        Path to the GeoTIFF file.
    """
    if provider is None:
        provider = cx.providers.Esri.WorldImagery

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate cache filename
    tile_hash = bbox_hash(lon_min, lat_min, lon_max, lat_max, zoom)
    output_path = str(cache_path / f"terrain_{tile_hash}_z{zoom}.tif")

    if os.path.exists(output_path):
        print(f"[TileLoader] Using cached GeoTIFF: {output_path}")
        return output_path

    print(f"[TileLoader] Downloading tiles for bbox: "
          f"({lat_min:.6f}, {lon_min:.6f}) -> ({lat_max:.6f}, {lon_max:.6f}), zoom={zoom}")
    print(f"[TileLoader] Provider: {provider.get('name', 'Esri.WorldImagery')}")

    num_tiles = 0
    try:
        num_tiles = cx.howmany(lon_min, lat_min, lon_max, lat_max, zoom, ll=True)
    except Exception:
        pass

    # Uncompressed memory estimate: num_tiles * 256 * 256 pixels * 3 bytes (RGB)
    estimated_size_mb = (num_tiles * 256 * 256 * 3) / (1024 * 1024)
    print(f"[TileLoader] Estimated tiles: {num_tiles} (~{estimated_size_mb:.1f} MB uncompressed in RAM)")

    tracker = ProgressTracker(num_tiles)
    
    # Monkey-patch contextily to track progress
    original_fetch_tile = getattr(cx.tile, '_fetch_tile', None)
    if original_fetch_tile:
        def patched_fetch_tile(*args, **kwargs):
            res = original_fetch_tile(*args, **kwargs)
            tracker.downloaded += 1
            return res
        cx.tile._fetch_tile = patched_fetch_tile

    tracker.start()

    try:
        # contextily expects (west, south, east, north) and ll=True for WGS84
        _ = cx.bounds2raster(
            lon_min,      # west
            lat_min,      # south
            lon_max,      # east
            lat_max,      # north
            output_path,
            zoom=zoom,
            source=provider,
            ll=True,      # input is lat/lon (WGS84)
        )
    finally:
        tracker.stop()
        if original_fetch_tile:
            cx.tile._fetch_tile = original_fetch_tile

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[TileLoader] GeoTIFF saved: {output_path} ({file_size_mb:.1f} MB)")

    return output_path


def download_elevation(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    zoom: int = 14,
    cache_dir: str = ".tile_cache",
) -> str:
    """
    Download AWS Terrain elevation tiles for the given bounding box.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Generate cache filename
    tile_hash = bbox_hash(lon_min, lat_min, lon_max, lat_max, zoom)
    output_path = str(cache_path / f"elevation_{tile_hash}_z{zoom}.tif")

    if os.path.exists(output_path):
        print(f"[TileLoader] Using cached Elevation GeoTIFF: {output_path}")
        return output_path

    print(f"[TileLoader] Downloading Elevation tiles for bbox: "
          f"({lat_min:.6f}, {lon_min:.6f}) -> ({lat_max:.6f}, {lon_max:.6f}), zoom={zoom}")

    num_tiles = 0
    try:
        num_tiles = cx.howmany(lon_min, lat_min, lon_max, lat_max, zoom, ll=True)
    except Exception:
        pass

    tracker = ProgressTracker(num_tiles)
    
    # Monkey-patch contextily to track progress
    original_fetch_tile = getattr(cx.tile, '_fetch_tile', None)
    if original_fetch_tile:
        def patched_fetch_tile(*args, **kwargs):
            res = original_fetch_tile(*args, **kwargs)
            tracker.downloaded += 1
            return res
        cx.tile._fetch_tile = patched_fetch_tile

    tracker.start()

    # Custom provider for AWS Terrarium
    # R * 256 + G + B / 256 - 32768
    aws_terrarium = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

    try:
        _ = cx.bounds2raster(
            lon_min,      # west
            lat_min,      # south
            lon_max,      # east
            lat_max,      # north
            output_path,
            zoom=zoom,
            source=aws_terrarium,
            ll=True,
        )
    finally:
        tracker.stop()
        if original_fetch_tile:
            cx.tile._fetch_tile = original_fetch_tile

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[TileLoader] Elevation GeoTIFF saved: {output_path} ({file_size_mb:.1f} MB)")

    return output_path
