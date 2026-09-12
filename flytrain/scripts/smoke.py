#!/usr/bin/env python3
"""End-to-end smoke on the synthetic graph. Run from the flytrain folder:

    python scripts/smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_pipeline import (
    test_constants_match_spec,
    test_end_to_end_train,
    test_hex_pixel_xy_unit,
    test_hex_rates_bright_patch_not_uniform,
    test_letterbox,
    test_synth_graph_has_three_cuts,
)


def main() -> None:
    import tempfile

    test_constants_match_spec()
    test_hex_pixel_xy_unit()
    test_letterbox()
    test_synth_graph_has_three_cuts()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_hex_rates_bright_patch_not_uniform(p)
        test_end_to_end_train(p)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
