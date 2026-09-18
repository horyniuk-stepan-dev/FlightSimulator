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

    def __init__(
        self,
        geotiff_path: str,
        elevation_path: str = None,
        elevation_format: str = "auto",
    ):
        """
        Load a GeoTIFF and prepare coordinate transforms.

        Args:
            geotiff_path: Path to GeoTIFF file (expected EPSG:3857).
            elevation_path: Path to Elevation GeoTIFF file.
            elevation_format: ``auto``, RGB ``terrarium``, or one-band
                ``meters``. Auto selects Terrarium for rasters with at least
                three bands and metres for single-band rasters.
        """
        self.path = geotiff_path
        self.elevation_path = elevation_path
        if elevation_format not in {"auto", "terrarium", "meters"}:
            raise ValueError("elevation_format must be auto, terrarium, or meters")
        self.elevation_format = elevation_format
        self.elevation_format_actual = None

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

        center_lon, center_lat = self._mercator_to_wgs84.transform(
            self._center_x, self._center_y
        )
        self.reference_gps = (float(center_lat), float(center_lon))
        # Web Mercator is conformal but its coordinate units are enlarged by
        # sec(latitude). Use a local tangent-scale approximation so flight
        # speed, camera footprint and altitude are all expressed in ground metres.
        self.ground_scale = float(np.cos(np.radians(center_lat)))

        # Compute resolution (meters per pixel)
        self.res_x = abs(self._transform.a)  # meters/pixel in X
        self.res_y = abs(self._transform.e)  # meters/pixel in Y (negative in affine)
        self.local_res_x = self.res_x * self.ground_scale
        self.local_res_y = self.res_y * self.ground_scale

        self.elevation = None
        self.base_elevation = 0.0
        self._elevation_transform = None
        self._elevation_inv_transform = None
        self._elevation_crs = None
        self._elevation_nodata = None
        self._map_to_elevation_crs = None
        
        if self.elevation_path:
            with rasterio.open(self.elevation_path) as src_elev:
                actual_format = elevation_format
                if actual_format == "auto":
                    actual_format = "terrarium" if src_elev.count >= 3 else "meters"
                if actual_format == "terrarium":
                    if src_elev.count < 3:
                        raise ValueError(
                            "Terrarium elevation requires at least three raster bands"
                        )
                    elev_data = src_elev.read([1, 2, 3], masked=True)
                    red = elev_data[0].astype(np.float32)
                    green = elev_data[1].astype(np.float32)
                    blue = elev_data[2].astype(np.float32)
                    decoded = (red * 256.0 + green + blue / 256.0) - 32768.0
                    self.elevation = np.asarray(
                        np.ma.filled(decoded, np.nan), dtype=np.float32
                    )
                else:
                    decoded = src_elev.read(1, masked=True).astype(np.float32)
                    self.elevation = np.asarray(
                        np.ma.filled(decoded, np.nan), dtype=np.float32
                    )
                self.elevation_format_actual = actual_format
                self._elevation_transform = src_elev.transform
                self._elevation_inv_transform = ~src_elev.transform
                self._elevation_crs = src_elev.crs
                self._elevation_nodata = src_elev.nodata
                if self._crs != self._elevation_crs:
                    self._map_to_elevation_crs = Transformer.from_crs(
                        self._crs, self._elevation_crs, always_xy=True
                    )
                finite = self.elevation[np.isfinite(self.elevation)]
                if finite.size == 0:
                    raise ValueError("Elevation raster contains no finite heights")
                self.base_elevation = float(np.min(finite))

        print(f"[OrthophotoMap] Loaded: {self.image.shape[1]}x{self.image.shape[0]} px, "
              f"resolution: {self.local_res_x:.3f} ground-m/px "
              f"({self.res_x:.3f} projected units/px)")
        print(f"[OrthophotoMap] Center (WM): ({self._center_x:.1f}, {self._center_y:.1f})")
        if self.elevation is not None:
            print(
                f"[OrthophotoMap] Elevation loaded ({self.elevation_format_actual}). "
                f"Min: {np.nanmin(self.elevation):.1f}m, "
                f"Max: {np.nanmax(self.elevation):.1f}m"
            )

    def generate_hillshade(
        self,
        azimuth_deg: float = 315.0,
        altitude_deg: float = 45.0,
        z_factor: float = 2.0,
    ) -> np.ndarray | None:
        """
        Compute hillshade illumination (0.0 to 1.0) from elevation raster.

        Args:
            azimuth_deg: Light source direction in degrees (315° = NW standard).
            altitude_deg: Light source angle above horizon (45° standard).
            z_factor: Vertical exaggeration factor for terrain slope.

        Returns:
            Hillshade map as float32 array [0.0, 1.0] matching elevation dimensions,
            or None if elevation data is missing.
        """
        if self.elevation is None:
            return None

        azimuth_rad = np.radians(360.0 - azimuth_deg + 90.0)
        altitude_rad = np.radians(altitude_deg)

        elev_h, elev_w = self.elevation.shape[:2]
        total_w_m, total_h_m = self.get_size_meters()
        dx_m = total_w_m / max(elev_w, 1)
        dy_m = total_h_m / max(elev_h, 1)

        dy, dx = np.gradient(self.elevation * z_factor)
        dz_dx = dx / dx_m
        dz_dy = -dy / dy_m

        slope = np.arctan(np.hypot(dz_dx, dz_dy))
        aspect = np.arctan2(dz_dy, -dz_dx)

        shaded = np.sin(altitude_rad) * np.cos(slope) + np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
        return np.clip(shaded, 0.0, 1.0).astype(np.float32)

    def apply_hillshade(
        self,
        blend_factor: float = 0.35,
        azimuth_deg: float = 315.0,
        altitude_deg: float = 45.0,
        z_factor: float = 2.0,
    ) -> None:
        """
        Blend 3D hillshade illumination into self.image to enhance terrain relief.

        Args:
            blend_factor: Shading intensity [0.0 = off, 1.0 = full].
            azimuth_deg: Light source direction angle.
            altitude_deg: Light source altitude angle.
            z_factor: Exaggeration multiplier for slopes.
        """
        hillshade = self.generate_hillshade(azimuth_deg, altitude_deg, z_factor)
        if hillshade is None:
            return

        h, w = self.image.shape[:2]
        hillshade_resized = cv2.resize(hillshade, (w, h), interpolation=cv2.INTER_LINEAR)
        hillshade_bgr = np.dstack([hillshade_resized] * 3)

        shade_multiplier = 0.5 + 0.8 * hillshade_bgr
        blended = self.image.astype(np.float32) * ((1.0 - blend_factor) + blend_factor * shade_multiplier)
        self.image = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        self.image = np.ascontiguousarray(self.image)
        print(f"[OrthophotoMap] Applied 3D Hillshade relief (blend={blend_factor:.2f}, z_factor={z_factor:.1f})")

    def apply_season(self, season: str, strength: float = 0.7) -> None:
        """
        Apply a synthetic seasonal look to self.image (see terrain/season.py).

        Args:
            season: "winter" (or "" / "summer" / "none" for no change).
            strength: 0.0 = off, 1.0 = maximum.
        """
        from simulator.terrain.season import apply_season as _apply

        key = (season or "").strip().lower()
        if key in ("", "none", "summer"):
            return
        self.image = np.ascontiguousarray(_apply(self.image, key, strength=strength))
        print(f"[OrthophotoMap] Applied synthetic season '{key}' (strength={strength:.2f})")

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
        ground_scale = getattr(self, "ground_scale", 1.0)
        x_min = (self._bounds.left - self._center_x) * ground_scale
        x_max = (self._bounds.right - self._center_x) * ground_scale
        y_min = (self._bounds.bottom - self._center_y) * ground_scale
        y_max = (self._bounds.top - self._center_y) * ground_scale
        return x_min, y_min, x_max, y_max

    def get_size_meters(self) -> tuple[float, float]:
        """Returns (width_m, height_m) of the map in meters."""
        ground_scale = getattr(self, "ground_scale", 1.0)
        return (
            (self._bounds.right - self._bounds.left) * ground_scale,
            (self._bounds.top - self._bounds.bottom) * ground_scale,
        )

    def local_to_pixel(self, lx: float, ly: float) -> tuple[float, float]:
        """
        Convert local coordinates (meters from map center) to pixel coordinates.

        Args:
            lx, ly: Local position in meters (x=east, y=north).

        Returns:
            (px, py): Pixel coordinates (column, row).
        """
        # Local → Web Mercator
        ground_scale = getattr(self, "ground_scale", 1.0)
        mx = lx / ground_scale + self._center_x
        my = ly / ground_scale + self._center_y
        # Web Mercator → pixel
        col, row = self._inv_transform * (mx, my)
        return float(col), float(row)

    def map_pixels_to_elevation_pixels(self, cols, rows):
        """Map orthophoto pixels to DEM pixels through both rasters' georeferencing."""
        if self._elevation_inv_transform is None:
            raise ValueError("Elevation raster is not available")
        cols = np.asarray(cols, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)
        t = self._transform
        xs = t.a * cols + t.b * rows + t.c
        ys = t.d * cols + t.e * rows + t.f
        if self._map_to_elevation_crs is not None:
            xs, ys = self._map_to_elevation_crs.transform(xs, ys)
        inv = self._elevation_inv_transform
        dem_cols = inv.a * xs + inv.b * ys + inv.c
        dem_rows = inv.d * xs + inv.e * ys + inv.f
        return np.asarray(dem_cols), np.asarray(dem_rows)

    def sample_elevation_at_map_pixels(self, cols, rows, *, return_valid=False):
        """Bilinearly sample DEM by orthophoto pixel coordinates.

        Out-of-coverage samples are NaN. They are never silently clamped to the
        nearest DEM edge because that would give a plausible height for the
        wrong geographic point.
        """
        if self.elevation is None:
            values = np.zeros_like(np.asarray(cols, dtype=np.float64))
            valid = np.ones_like(values, dtype=bool)
            return (values, valid) if return_valid else values
        from simulator.terrain.parallax import sample_bilinear

        dem_cols, dem_rows = self.map_pixels_to_elevation_pixels(cols, rows)
        h, w = self.elevation.shape[:2]
        valid = (
            np.isfinite(dem_cols)
            & np.isfinite(dem_rows)
            & (dem_cols >= 0.0)
            & (dem_cols <= w - 1.0)
            & (dem_rows >= 0.0)
            & (dem_rows <= h - 1.0)
        )
        safe_cols = np.where(valid, dem_cols, 0.0)
        safe_rows = np.where(valid, dem_rows, 0.0)
        values = sample_bilinear(self.elevation, safe_rows, safe_cols)
        valid &= np.isfinite(values)
        values = np.where(valid, values, np.nan)
        return (values, valid) if return_valid else values

    def map_to_elevation_pixel_affine(self) -> np.ndarray:
        """Return a 3x3 orthophoto-pixel -> DEM-pixel affine for GPU sampling."""
        if self._elevation_inv_transform is None:
            raise ValueError("Elevation raster is not available")
        if self._map_to_elevation_crs is not None:
            raise ValueError("GPU elevation sampling requires matching raster CRS")

        def matrix(affine):
            return np.array(
                [[affine.a, affine.b, affine.c],
                 [affine.d, affine.e, affine.f],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

        return matrix(self._elevation_inv_transform) @ matrix(self._transform)

    def pixel_to_local(self, px: float, py: float) -> tuple[float, float]:
        """
        Convert pixel coordinates to local coordinates (meters from map center).

        Args:
            px, py: Pixel coordinates (column, row).

        Returns:
            (lx, ly): Local position in meters.
        """
        mx, my = self._transform * (px, py)
        ground_scale = getattr(self, "ground_scale", 1.0)
        return (
            float((mx - self._center_x) * ground_scale),
            float((my - self._center_y) * ground_scale),
        )

    def local_to_mercator(self, lx: float, ly: float) -> tuple[float, float]:
        ground_scale = getattr(self, "ground_scale", 1.0)
        return (
            float(lx / ground_scale + self._center_x),
            float(ly / ground_scale + self._center_y),
        )

    def mercator_to_local(self, mx: float, my: float) -> tuple[float, float]:
        ground_scale = getattr(self, "ground_scale", 1.0)
        return (
            float((mx - self._center_x) * ground_scale),
            float((my - self._center_y) * ground_scale),
        )

    def local_to_gps(self, lx: float, ly: float) -> tuple[float, float]:
        """
        Convert local coordinates to GPS (WGS84 lat/lon).

        Returns:
            (lat, lon)
        """
        mx, my = self.local_to_mercator(lx, ly)
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
