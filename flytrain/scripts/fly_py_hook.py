"""Snippet to paste into fly.py after `W = build_W(nodes, edges).tocsc()`.

Keeps GAIN = 0.2. Only multiplies existing edges in the three allowed cuts.
"""

from pathlib import Path

# --- paste inside main(), after W is built ---------------------------------
_GAINS = Path(r"C:\Dev\flywow\flytrain\train_out\gains_type.csv")
if _GAINS.exists():
    import sys

    sys.path.append(r"C:\Dev\flywow\flytrain")
    from flytrain.bake import apply_type_gains

    W = apply_type_gains(W, nodes, _GAINS).tocsc()
    print(f"loaded type-gains {_GAINS}")
# ---------------------------------------------------------------------------
