"""
OrthophotoMap — wraps a GeoTIFF into a numpy array with geo-transform.
Provides pixel ↔ world (Web Mercator meters) coordinate conversion.
"""
import cv2
import numpy as np
import rasterio
from pyproj import Transformer


class OrthophotoMap:
    """
    Manages the orthophoto raster data and coordinate transformations.

    The internal coordinate system is Web Mercator (EPSG:3857) in meters.
    The "local" coordinate system is offset so that the map center is at (0, 0).
    """

    def __init__(self, geotiff_path: str, elevation_path: str = None):
        """
        Load a GeoTIFF and prepare coordinate transforms.

        Args:
            geotiff_path: Path to GeoTIFF file (expected EPSG:3857).
            elevation_path: Path to Elevation GeoTIFF file.
        """
        self.path = geotiff_path
        self.elevation_path = elevation_path

        with rasterio.open(geotiff_path) as src:
            # Read as (bands, H, W), then transpose to (H, W, bands) for OpenCV
            data = src.read()  # shape: (bands, H, W)
            self.image = np.transpose(data, (1, 2, 0))  # (H, W, C)

            # Convert RGBA to BGR if needed, or RGB to BGR
            if self.image.shape[2] == 4:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGBA2BGR)
            elif self.image.shape[2] == 3:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

            self.image = np.ascontiguousarray(self.image)

            # Affine transform: pixel → Web Mercator meters
            self._transform = src.transform
            self._inv_transform = ~src.transform  # inverse: meters → pixel
            self._crs = src.crs
            self._bounds = src.bounds  # BoundingBox(left, bottom, right, top)

        # Compute map center in Web Mercator meters (for local coordinate system)
        self._center_x = (self._bounds.left + self._bounds.right) / 2.0
        self._center_y = (self._bounds.bottom + self._bounds.top) / 2.0

        # Transformer for WGS84 ↔ Web Mercator
        self._wgs84_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self._mercator_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # Compute resolution (meters per pixel)
        self.res_x = abs(self._transform.a)  # meters/pixel in X
        self.res_y = abs(self._transform.e)  # meters/pixel in Y (negative in affine)

        self.elevation = None
        self.base_elevation = 0.0
        
        if self.elevation_path:
            with rasterio.open(self.elevation_path) as src_elev:
                # Read RGB bands
                elev_data = src_elev.read()
                R = elev_data[0].astype(np.float32)
                G = elev_data[1].astype(np.float32)
                B = elev_data[2].astype(np.float32)
                
                # AWS Terrarium formula
                self.elevation = (R * 256.0 + G + B / 256.0) - 32768.0
                self.base_elevation = np.min(self.elevation)

        print(f"[OrthophotoMap] Loaded: {self.image.shape[1]}x{self.image.shape[0]} px, "
              f"resolution: {self.res_x:.3f} m/px")
        print(f"[OrthophotoMap] Center (WM): ({self._center_x:.1f}, {self._center_y:.1f})")
        if self.elevation is not None:
            print(f"[OrthophotoMap] Elevation loaded. Min: {np.min(self.elevation):.1f}m, Max: {np.max(self.elevation):.1f}m")

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    def get_bounds_local(self) -> tuple[float, float, float, float]:
        """
        Get map bounds in local coordinates (meters, centered at map center).
        Returns: (x_min, y_min, x_max, y_max)
        """
        x_min = self._bounds.left - self._center_x
        x_max = self._bounds.right - self._center_x
        y_min = self._bounds.bottom - self._center_y
        y_max = self._bounds.top - self._center_y
        return x_min, y_min, x_max, y_max

    def get_size_meters(self) -> tuple[float, float]:
        """Returns (width_m, height_m) of the map in meters."""
        return (self._bounds.right - self._bounds.left,
                self._bounds.top - self._bounds.bottom)

    def local_to_pixel(self, lx: float, ly: float) -> tuple[float, float]:
        """
        Convert local coordinates (meters from map center) to pixel coordinates.

        Args:
            lx, ly: Local position in meters (x=east, y=north).

        Returns:
            (px, py): Pixel coordinates (column, row).
        """
        # Local → Web Mercator
        mx = lx + self._center_x
        my = ly + self._center_y
        # Web Mercator → pixel
        col, row = self._inv_transform * (mx, my)
        return float(col), float(row)

    def pixel_to_local(self, px: float, py: float) -> tuple[float, float]:
        """
        Convert pixel coordinates to local coordinates (meters from map center).

        Args:
            px, py: Pixel coordinates (column, row).

        Returns:
            (lx, ly): Local position in meters.
        """
        mx, my = self._transform * (px, py)
        return float(mx - self._center_x), float(my - self._center_y)

    def local_to_gps(self, lx: float, ly: float) -> tuple[float, float]:
        """
        Convert local coordinates to GPS (WGS84 lat/lon).

        Returns:
            (lat, lon)
        """
        mx = lx + self._center_x
        my = ly + self._center_y
        lon, lat = self._mercator_to_wgs84.transform(mx, my)
        return lat, lon

    def gps_to_local(self, lat: float, lon: float) -> tuple[float, float]:
        """
        Convert GPS (WGS84 lat/lon) to local coordinates.

        Returns:
            (lx, ly): Local position in meters.
        """
        mx, my = self._wgs84_to_mercator.transform(lon, lat)
        return float(mx - self._center_x), float(my - self._center_y)
