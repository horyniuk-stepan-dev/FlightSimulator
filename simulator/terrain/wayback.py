"""
Esri Wayback — вибір епохи (дати) супутникових знімків для карти симулятора.

Esri World Imagery публікує "релізи" мозаїки: кожен реліз — знімок стану
шару на певну дату. Wayback дозволяє завантажити тайли будь-якого релізу,
тобто зняти той самий район у різні періоди.

ВАЖЛИВО: дата релізу != дата зйомки. Мозаїка оновлюється для конкретного
району лише тоді, коли його переznімали. Щоб дізнатись, які релізи реально
відрізняються для вашої точки, використайте CLI:

    python -m simulator.terrain.wayback --lat 50.45 --lon 30.52
"""

from __future__ import annotations

import argparse
import json
import math
import re
from urllib.error import URLError
from dataclasses import dataclass
from datetime import date, timedelta
from urllib.request import urlopen

WAYBACK_CONFIG_URL = (
    "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
)
_TILE_BASE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
    "WMTS/1.0.0/default028mm/MapServer"
)
_TITLE_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

_RELEASES_CACHE: list["WaybackRelease"] | None = None


@dataclass(frozen=True)
class WaybackRelease:
    """Один реліз Wayback: ідентифікатор + дата публікації."""

    release_id: int
    release_date: date
    title: str

    @property
    def tile_url(self) -> str:
        """URL-шаблон у форматі contextily ({z}/{y}/{x} = level/row/col)."""
        return f"{_TILE_BASE}/tile/{self.release_id}/{{z}}/{{y}}/{{x}}"

    def __str__(self) -> str:
        return f"{self.release_date.isoformat()} (release {self.release_id})"


def _get_json(url: str, timeout: float = 30.0):
    with urlopen(url, timeout=timeout) as resp:  # noqa: S310 — фіксований хост Esri
        return json.loads(resp.read().decode("utf-8"))


def fetch_releases(timeout: float = 30.0, use_cache: bool = True) -> list[WaybackRelease]:
    """Повертає всі релізи Wayback, відсортовані від найновішого до найстарішого."""
    global _RELEASES_CACHE
    if use_cache and _RELEASES_CACHE is not None:
        return _RELEASES_CACHE

    cfg = _get_json(WAYBACK_CONFIG_URL, timeout=timeout)
    releases: list[WaybackRelease] = []
    for key, item in cfg.items():
        title = item.get("itemTitle", "")
        m = _TITLE_DATE_RE.search(title)
        if not m:
            continue
        y, mo, d = (int(g) for g in m.groups())
        releases.append(WaybackRelease(int(key), date(y, mo, d), title))

    releases.sort(key=lambda r: r.release_date, reverse=True)
    if use_cache:
        _RELEASES_CACHE = releases
    return releases


def _parse_map_date(map_date: str) -> date:
    """'2019' -> 2019-12-31, '2019-04' -> 2019-04-30, '2019-04-15' -> як є."""
    s = map_date.strip()
    parts = s.split("-")
    try:
        if len(parts) == 1:
            return date(int(parts[0]), 12, 31)
        if len(parts) == 2:
            y, mo = int(parts[0]), int(parts[1])
            first_next = date(y + 1, 1, 1) if mo == 12 else date(y, mo + 1, 1)
            return first_next - timedelta(days=1)
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise ValueError(f"Некоректна дата карти: {map_date!r}") from exc
    raise ValueError(f"Некоректна дата карти: {map_date!r} (очікується YYYY[-MM[-DD]])")


def resolve_release(
    map_date: str, releases: list[WaybackRelease] | None = None
) -> WaybackRelease:
    """Найновіший реліз, опублікований НЕ ПІЗНІШЕ за вказану дату."""
    target = _parse_map_date(map_date)
    releases = releases if releases is not None else fetch_releases()
    for rel in releases:  # відсортовані від найновішого
        if rel.release_date <= target:
            return rel
    oldest = releases[-1]
    raise ValueError(
        f"Немає релізу Wayback на {target.isoformat()}; "
        f"найстаріший доступний — {oldest.release_date.isoformat()}"
    )


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """WGS84 -> (col, row) = (x, y) тайла Web Mercator."""
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def versions_with_local_changes(
    lat: float,
    lon: float,
    zoom: int = 13,
    releases: list[WaybackRelease] | None = None,
    timeout: float = 30.0,
) -> list[WaybackRelease]:
    """
    Релізи, у яких знімок для заданої точки РЕАЛЬНО відрізняється.

    Використовує tilemap-ендпойнт: для релізу R він повертає "select":[S] —
    ідентифікатор релізу, чий тайл фактично віддається. Усі релізи між R і S
    містять той самий знімок, тому їх можна пропустити.
    """
    releases = releases if releases is not None else fetch_releases(timeout=timeout)
    index_of = {rel.release_id: i for i, rel in enumerate(releases)}
    col, row = lonlat_to_tile(lon, lat, zoom)

    found: list[WaybackRelease] = []
    i = 0
    while i < len(releases):
        rel = releases[i]
        url = f"{_TILE_BASE}/tilemap/{rel.release_id}/{zoom}/{row}/{col}"
        try:
            data = _get_json(url, timeout=timeout)
        except Exception:  # noqa: BLE001 — недоступний реліз просто пропускаємо
            i += 1
            continue

        select = data.get("select") or []
        if not select:
            i += 1
            continue

        src_id = int(select[0])
        src_idx = index_of.get(src_id)
        if src_idx is None:
            i += 1
            continue

        found.append(releases[src_idx])
        i = src_idx + 1  # усі проміжні релізи — той самий знімок

    return found


def fetch_tile_png(release: "WaybackRelease", col: int, row: int, zoom: int,
                   timeout: float = 30.0) -> bytes | None:
    """Сирі байти одного тайла релізу (None, якщо тайл недоступний)."""
    url = release.tile_url.format(z=zoom, y=row, x=col)
    try:
        with urlopen(url, timeout=timeout) as resp:  # noqa: S310 — фіксований хост Esri
            return resp.read()
    except (URLError, OSError):
        return None


def build_preview(
    lat: float,
    lon: float,
    out_path: str,
    zoom: int = 14,
    releases: list["WaybackRelease"] | None = None,
    cols: int = 5,
) -> str:
    """
    Контактний аркуш: по одному тайлу з кожної епохи, підписаний датою.

    Дозволяє оком вибрати літню/зимову епоху — дата релізу сезону не показує.
    """
    import cv2
    import numpy as np

    versions = releases if releases is not None else versions_with_local_changes(
        lat, lon, zoom=min(zoom, 13)
    )
    col, row = lonlat_to_tile(lon, lat, zoom)

    cell = 256
    label_h = 26
    tiles: list[tuple[str, "np.ndarray"]] = []
    for rel in versions:
        raw = fetch_tile_png(rel, col, row, zoom)
        if raw is None:
            print(f"  [skip] {rel.release_date.isoformat()} — тайл недоступний на zoom {zoom}")
            continue
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] {rel.release_date.isoformat()} — не вдалось декодувати")
            continue
        if img.shape[0] != cell or img.shape[1] != cell:
            img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        tiles.append((rel.release_date.isoformat(), img))
        print(f"  [ok]   {rel.release_date.isoformat()}")

    if not tiles:
        raise RuntimeError("Жодного тайла не завантажено — спробуйте менший --zoom")

    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (cell + label_h), cols * cell, 3), 32, dtype=np.uint8)
    for i, (label, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = r * (cell + label_h)
        x0 = c * cell
        sheet[y0 + label_h : y0 + label_h + cell, x0 : x0 + cell] = img
        cv2.putText(sheet, label, (x0 + 6, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, sheet)
    print(f"[Wayback] Прев'ю збережено: {out_path} ({len(tiles)} епох, zoom {zoom})")
    return out_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Показати епохи знімків Esri Wayback, доступні для точки"
    )
    parser.add_argument("--lat", type=float, required=True, help="Широта точки")
    parser.add_argument("--lon", type=float, required=True, help="Довгота точки")
    parser.add_argument(
        "--zoom",
        type=int,
        default=13,
        help="Zoom для перевірки (13 ≈ 3 км/тайл; більший zoom — точніше, але повільніше)",
    )
    parser.add_argument("--all", action="store_true", help="Показати ВСІ релізи, без фільтра")
    parser.add_argument(
        "--preview",
        type=str,
        default="",
        help="Шлях до PNG: контактний аркуш з одним тайлом кожної епохи "
        "(щоб оком вибрати літню/зимову)",
    )
    parser.add_argument(
        "--preview-zoom", type=int, default=14, help="Zoom тайлів у прев'ю (типово 14)"
    )
    args = parser.parse_args()

    releases = fetch_releases()
    print(f"[Wayback] Усього релізів: {len(releases)}")

    if args.all:
        for rel in releases:
            print(f"  {rel.release_date.isoformat()}  id={rel.release_id}")
        return

    print(f"[Wayback] Пошук версій зі змінами для ({args.lat:.5f}, {args.lon:.5f})...")
    versions = versions_with_local_changes(args.lat, args.lon, zoom=args.zoom)
    print(f"[Wayback] Різних знімків для цієї точки: {len(versions)}")
    for rel in versions:
        print(f"  --map-date {rel.release_date.isoformat()}   (release {rel.release_id})")

    if args.preview:
        print(f"[Wayback] Завантаження прев'ю (zoom {args.preview_zoom})...")
        build_preview(
            args.lat, args.lon, args.preview, zoom=args.preview_zoom, releases=versions
        )


if __name__ == "__main__":
    _main()
