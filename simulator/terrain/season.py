"""
Синтетична сезонна трансформація ортофото.

Esri World Imagery навмисно відбирає безхмарні, безсніжні, leaf-on знімки,
тому справжньої зими для більшості районів у ній немає. Цей модуль створює
КЕРОВАНИЙ фотометричний зсув, що імітує зимовий вигляд.

Ключ до правдоподібності — розділити «зелене й гладке» (поле → під снігом,
біле) і «зелене й шорстке» (крона дерев → темна, безлиста). Самого лише
excess-green не досить: на реальному знімку z17 понад половина пікселів
"зелені", і фільтр по ньому одному робить рівномірний сірий туман.

  1. tree = excess_green × локальна_шорсткість → темніє;
  2. десатурація майже до сірого;
  3. сніг — усе, що не крона, тягнеться до білого, але темні пікселі
     (асфальт, вода, тіні) лишаються темними;
  4. мікродеталь повертається через high-pass, щоб борозни, межі полів
     і дороги лишались придатними для матчингу;
  5. холодний тон.

ЦЕ НЕ СПРАВЖНЯ ЗЙОМКА. Геометрія сцени й напрям тіней не змінюються —
змінюється лише фотометрика, тобто саме те, що ламає дескриптори.
"""

from __future__ import annotations

import cv2
import numpy as np

SEASONS = ("winter",)

# Емпіричні константи (фіксовані, не залежать від блока — інакше будуть шви)
_EXG_FULL = 0.28  # excess-green, за якого поверхня вважається повністю "живою"
_TEXTURE_FULL = 14.0  # локальна варіація (рівні сірого), за якої поверхня "шорстка"
_TEXTURE_KSIZE = 7
_DETAIL_SIGMA = 2.0  # high-pass для повернення мікродеталі
_DETAIL_GAIN = 0.55
_SNOW_BGR = (252.0, 250.0, 246.0)  # холодний білий (BGR: більше синього)


def _winterize_block(blk_bgr: np.ndarray, s: float) -> np.ndarray:
    """Зимовий фільтр для одного блока (H, W, 3) float32 BGR у 0..255."""
    b, g, r = blk_bgr[..., 0], blk_bgr[..., 1], blk_bgr[..., 2]
    total = b + g + r + 1e-6
    exg = np.clip((2.0 * g - r - b) / total, 0.0, None)
    veg = np.clip(exg / _EXG_FULL, 0.0, 1.0)
    lum = 0.114 * b + 0.587 * g + 0.299 * r

    # крона = зелене + шорстке (поле зелене, але гладке → під снігом)
    k = (_TEXTURE_KSIZE, _TEXTURE_KSIZE)
    texture = cv2.blur(np.abs(lum - cv2.blur(lum, k)), k)
    rough = np.clip(texture / _TEXTURE_FULL, 0.0, 1.0)
    tree = veg * rough

    # 1) десатурація
    desat = 0.90 * s
    b = b + (lum - b) * desat
    g = g + (lum - g) * desat
    r = r + (lum - r) * desat

    # 2) безлиста крона темніє
    canopy = 1.0 - 0.40 * s * tree
    b *= canopy
    g *= canopy
    r *= canopy
    lum_c = lum * canopy

    # 3) сніг на всьому, що не крона; дуже темне (асфальт, вода, тінь) лишається
    darkfac = np.clip((lum_c - 25.0) / 45.0, 0.0, 1.0)
    snow = 0.92 * s * (1.0 - tree) * darkfac
    b = b * (1.0 - snow) + _SNOW_BGR[0] * snow
    g = g * (1.0 - snow) + _SNOW_BGR[1] * snow
    r = r * (1.0 - snow) + _SNOW_BGR[2] * snow

    # 4) повернення мікродеталі (борозни, межі, дороги лишаються матчабельними)
    high_pass = (lum - cv2.GaussianBlur(lum, (0, 0), _DETAIL_SIGMA)) * _DETAIL_GAIN
    b += high_pass
    g += high_pass
    r += high_pass

    # 5) холодний тон
    b += 5.0 * s
    r -= 4.0 * s

    return np.clip(np.stack((b, g, r), axis=-1), 0.0, 255.0)


def winterize(
    image_bgr: np.ndarray,
    strength: float = 0.85,
    block_rows: int = 1024,
    halo: int = 16,
) -> np.ndarray:
    """
    Зимовий фільтр для всієї карти, блоками (карта буває ~9k x 9k px).

    halo — перекриття блоків, щоб box/Gaussian-фільтри не давали швів.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Очікується BGR (H, W, 3), отримано {image_bgr.shape}")
    s = float(np.clip(strength, 0.0, 1.0))
    if s == 0.0:
        return image_bgr

    h = image_bgr.shape[0]
    out = np.empty_like(image_bgr)
    for y0 in range(0, h, block_rows):
        y1 = min(h, y0 + block_rows)
        ys = max(0, y0 - halo)
        ye = min(h, y1 + halo)
        res = _winterize_block(image_bgr[ys:ye].astype(np.float32), s)
        out[y0:y1] = res[y0 - ys : y0 - ys + (y1 - y0)].astype(np.uint8)
    return out


def apply_season(image_bgr: np.ndarray, season: str, strength: float = 0.85) -> np.ndarray:
    """Диспетчер сезонів. Порожній рядок / 'summer' / 'none' — без змін."""
    key = (season or "").strip().lower()
    if key in ("", "none", "summer"):
        return image_bgr
    if key == "winter":
        return winterize(image_bgr, strength=strength)
    raise ValueError(f"Невідомий сезон: {season!r} (доступні: {', '.join(SEASONS)})")
