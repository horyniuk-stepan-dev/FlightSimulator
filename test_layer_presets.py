"""run_parallel_bochkivtsi.py presets: the default reproduces the testtopboch_max bench recipe.

Run: python -m pytest -q test_layer_presets.py
"""

import run_parallel_bochkivtsi as rpb


def tasks(*argv):
    args = rpb.parse_args(list(argv))
    return args, {t["altitude"]: t for t in rpb.build_tasks(args)}


def test_default_is_the_max_bench_recipe():
    args, by_alt = tasks()
    assert args.preset == "max" and args.heading_hold_deg == 0.0
    expected = {
        1000: ("bochkivtsi_max_1000m", 100.0, 18, 5),
        2000: ("bochkivtsi_max_2000m", 150.0, 17, 3),
        500: ("bochkivtsi_max25_500m", 25.0, 18, 20),
    }
    for altitude, (name, speed, zoom, anchor_slots) in expected.items():
        task = by_alt[altitude]
        assert task["name"] == name and task["output_dir"].name == name
        assert (task["speed"], task["zoom"], task["anchor_slots"]) == (speed, zoom, anchor_slots)


def test_original_preset_and_overrides():
    args, by_alt = tasks("--preset", "original")
    assert args.heading_hold_deg is None
    assert by_alt[1000]["name"] == "bochkivtsi" and by_alt[1000]["speed"] == 150.0
    assert by_alt[1000]["zoom"] is None and by_alt[1000]["anchor_slots"] is None

    args, by_alt = tasks("--only", "500", "--speed", "500=50", "--tag", "x", "--heading-hold-deg", "-1")
    assert list(by_alt) == [500] and args.heading_hold_deg is None
    task = by_alt[500]
    assert task["name"] == "bochkivtsi_x_500m" and task["speed"] == 50.0 and task["zoom"] == 18
    assert task["anchor_slots"] == 10

    _, by_alt = tasks("--anchor-spacing-m", "0")
    assert by_alt[1000]["anchor_slots"] is None
