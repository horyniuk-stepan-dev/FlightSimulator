"""
FPV drone component catalog + builder.

A build = frame + 4 motors + 4 props + battery + electronics. The builder turns
the chosen parts into the physical numbers the simulator needs (mass, inertia,
motor/prop/battery coefficients), so swapping a part changes flight behaviour.

All catalog numbers are approximate, typical-of-class values (not datasheet
copies). Prop Ct/Cp are rough estimates derived from pitch/diameter; tweak them
if you have thrust-stand data.

Prop convention: T = Ct * rho * n^2 * D^4,  Q = Cp/(2*pi) * rho * n^2 * D^5
(n in rev/s, D in metres).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

INCH = 0.0254
RHO_AIR = 1.225


@dataclass(frozen=True)
class Frame:
    name: str
    wheelbase_mm: float  # motor-to-motor diagonal
    mass_g: float
    max_prop_in: float
    drag_area_m2: float  # effective frontal C_d*A of the whole craft


@dataclass(frozen=True)
class Motor:
    name: str
    kv: float  # rpm per volt
    stator: str
    mass_g: float
    resistance_ohm: float  # phase-to-phase winding resistance
    no_load_current_a: float
    max_current_a: float


@dataclass(frozen=True)
class Prop:
    name: str
    diameter_in: float
    pitch_in: float
    blades: int
    mass_g: float

    @property
    def ct(self) -> float:
        pd = self.pitch_in / self.diameter_in
        blade_f = {2: 0.80, 3: 1.0, 4: 1.12}.get(self.blades, 1.0)
        return (0.07 + 0.09 * pd) * blade_f

    @property
    def cp(self) -> float:
        pd = self.pitch_in / self.diameter_in
        blade_f = {2: 0.78, 3: 1.0, 4: 1.15}.get(self.blades, 1.0)
        return (0.03 + 0.085 * pd**1.5) * blade_f  # ~FM 0.4, typical small FPV prop


@dataclass(frozen=True)
class Battery:
    name: str
    cells: int
    capacity_mah: float
    c_rating: float
    mass_g: float
    cell_resistance_ohm: float  # per-cell internal resistance (fresh pack)


@dataclass(frozen=True)
class Electronics:
    """FC/ESC stack, camera, VTX, receiver, GPS, mounts, straps."""

    name: str
    mass_g: float
    esc_max_current_a: float
    motor_idle: float = 0.055  # Betaflight dshot_idle_value 5.5 %


CATALOG: dict[str, dict] = {
    "frame": {
        "3in_cinewhoop": Frame("3in cinewhoop (ducted)", 140, 95, 3.0, 0.030),
        "5in_freestyle": Frame("5in freestyle 220mm", 220, 120, 5.1, 0.018),
        "5in_race": Frame("5in race 210mm (light)", 210, 85, 5.1, 0.014),
        "7in_longrange": Frame("7in long range 300mm", 300, 165, 7.5, 0.024),
    },
    "motor": {
        "1404_4600kv": Motor("1404 4600KV", 4600, "1404", 9.5, 0.200, 0.40, 15),
        "2207_1950kv": Motor("2207 1950KV", 1950, "2207", 32, 0.065, 1.10, 45),
        "2207_2550kv": Motor("2207 2550KV", 2550, "2207", 32, 0.045, 1.40, 50),
        "2306_2450kv": Motor("2306 2450KV", 2450, "2306", 31, 0.055, 1.20, 42),
        "2806_1300kv": Motor("2806.5 1300KV", 1300, "2806", 46, 0.085, 0.80, 40),
    },
    "prop": {
        "3x3_tri": Prop("3x3 tri", 3.0, 3.0, 3, 1.6),
        "5x4.3_tri": Prop("5x4.3 tri", 5.1, 4.3, 3, 4.3),
        "5x3_tri": Prop("5x3 tri (efficient)", 5.1, 3.0, 3, 3.9),
        "5x5_tri": Prop("5x5 tri (aggressive)", 5.1, 5.0, 3, 4.6),
        "7x4_bi": Prop("7x4 bi", 7.0, 4.0, 2, 6.5),
        "7x3.5_tri": Prop("7x3.5 tri", 7.0, 3.5, 3, 8.8),
    },
    "battery": {
        "4s_650": Battery("4S 650mAh 75C", 4, 650, 75, 80, 0.012),
        "4s_1500": Battery("4S 1500mAh 100C", 4, 1500, 100, 180, 0.006),
        "6s_1100": Battery("6S 1100mAh 120C", 6, 1100, 120, 185, 0.007),
        "6s_1300": Battery("6S 1300mAh 120C", 6, 1300, 120, 215, 0.006),
        "6s_3000_liion": Battery("6S 3000mAh Li-ion 21700", 6, 3000, 5, 300, 0.020),
        "6s_4000_liion": Battery("6S 4000mAh Li-ion 21700", 6, 4000, 5, 400, 0.020),
    },
    "electronics": {
        "whoop_stack": Electronics("AIO 20A + analog cam/VTX", 35, 20),
        "freestyle_stack": Electronics("30x30 FC + 55A ESC + digital VTX", 95, 55),
        "race_stack": Electronics("20x20 FC + 45A ESC + analog VTX", 55, 45),
        "lr_stack_gps": Electronics("FC + 45A ESC + digital VTX + GPS", 120, 45),
    },
}

PRESETS: dict[str, dict[str, str]] = {
    "cinewhoop_3in": {
        "frame": "3in_cinewhoop",
        "motor": "1404_4600kv",
        "prop": "3x3_tri",
        "battery": "4s_650",
        "electronics": "whoop_stack",
    },
    "freestyle_5in_6s": {
        "frame": "5in_freestyle",
        "motor": "2207_1950kv",
        "prop": "5x4.3_tri",
        "battery": "6s_1300",
        "electronics": "freestyle_stack",
    },
    "race_5in_6s": {
        "frame": "5in_race",
        "motor": "2207_2550kv",
        "prop": "5x5_tri",
        "battery": "6s_1100",
        "electronics": "race_stack",
    },
    "longrange_7in_6s": {
        "frame": "7in_longrange",
        "motor": "2806_1300kv",
        "prop": "7x3.5_tri",
        "battery": "6s_4000_liion",
        "electronics": "lr_stack_gps",
    },
}


@dataclass
class DroneBuild:
    frame: Frame
    motor: Motor
    prop: Prop
    battery: Battery
    electronics: Electronics
    payload_g: float = 0.0
    warnings: list[str] = field(default_factory=list)

    # ---- derived physical quantities ------------------------------------
    @property
    def arm_m(self) -> float:
        """Centre-to-motor distance."""
        return self.frame.wheelbase_mm / 2000.0

    @property
    def mass_kg(self) -> float:
        g = (
            self.frame.mass_g
            + 4 * (self.motor.mass_g + self.prop.mass_g)
            + self.battery.mass_g
            + self.electronics.mass_g
            + self.payload_g
        )
        return g / 1000.0

    @property
    def inertia(self) -> tuple[float, float, float]:
        """Ixx, Iyy, Izz [kg m^2]: motors as point masses on an X, frame as
        a thin plate, battery as a box on top of the centre."""
        a = self.arm_m / math.sqrt(2)  # x/y offset of each motor
        m_mp = (self.motor.mass_g + self.prop.mass_g) / 1000.0
        ixx = 4 * m_mp * a**2
        izz = 4 * m_mp * 2 * a**2
        m_f = self.frame.mass_g / 1000.0
        L = self.frame.wheelbase_mm / 1000.0
        ixx += m_f * L**2 / 24
        izz += m_f * L**2 / 12
        m_c = (self.battery.mass_g + self.electronics.mass_g + self.payload_g) / 1000.0
        # ~ 0.07 x 0.035 x 0.04 m box, offset 3 cm above CoG
        ixx += m_c * (0.035**2 + 0.04**2) / 12 + m_c * 0.03**2
        izz += m_c * (0.07**2 + 0.035**2) / 12
        return ixx, ixx, izz

    @property
    def prop_inertia(self) -> float:
        """Rotating inertia per motor (prop as rod + motor bell)."""
        r = self.prop.diameter_in * INCH / 2
        j_prop = (self.prop.mass_g / 1000.0) * r**2 / 3
        j_bell = 0.4 * (self.motor.mass_g / 1000.0) * (0.012 * self.motor.mass_g / 32) ** 2
        return j_prop + j_bell

    def to_dict(self) -> dict:
        return {k: asdict(getattr(self, k)) for k in ("frame", "motor", "prop", "battery", "electronics")} | {
            "payload_g": self.payload_g
        }


def build(frame: str, motor: str, prop: str, battery: str, electronics: str, payload_g: float = 0.0) -> DroneBuild:
    c = CATALOG
    b = DroneBuild(
        c["frame"][frame],
        c["motor"][motor],
        c["prop"][prop],
        c["battery"][battery],
        c["electronics"][electronics],
        payload_g,
    )
    if b.prop.diameter_in > b.frame.max_prop_in + 1e-6:
        b.warnings.append(f"prop {b.prop.diameter_in}in does not fit frame (max {b.frame.max_prop_in}in)")
    if b.motor.max_current_a > b.electronics.esc_max_current_a * 1.1:
        b.warnings.append("ESC rating is below motor max current — ESC current limit will cap thrust")
    return b


def build_preset(name: str, **overrides) -> DroneBuild:
    parts = dict(PRESETS[name])
    payload = overrides.pop("payload_g", 0.0)
    parts.update(overrides)
    return build(**parts, payload_g=payload)
