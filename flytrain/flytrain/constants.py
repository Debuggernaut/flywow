"""LIF + vision constants. Must match fly.py at deploy."""

from __future__ import annotations

MIN_WEIGHT = 5

TICK_HZ = 100
INNER_STEPS = 2
INNER_DT = 0.01 / INNER_STEPS  # 5 ms; spec writes 0.005
WINDOW_S = 0.100
WINDOW_INNER_STEPS = int(round(WINDOW_S / INNER_DT))  # 20

VISION_HZ_MAX = 80.0
FOV_PX = 256
# Pan the hex sample window right on the 256² luma FOV.
# 0.25 = 64 px. Hex mapping is left-packed when the lattice is taller than
# wide, which was dropping the right side of the screen.
FOV_SHIFT_X_FRAC = 0.18
FOV_SHIFT_Y_FRAC = 0.0
DRIVE_PR = False
DRIVE_VPN = True
DRIVE_GRN = False

TAU_M = 0.020
TAU_S = 0.005
VREST = -52.0
VTH = -45.0
VRESET = -52.0
TREF = 0.0022
WSYN = 0.275
GAIN = 0.2

FAST_NT = {
    "acetylcholine": 1,
    "dopamine": 1,
    "octopamine": 1,
    "serotonin": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
}

VISUAL_SUPERCLASSES = (
    "ol_sensory",
    "ol_intrinsic",
    "visual_projection",
    "visual_centrifugal",
)

# Three trainable cuts (Tier A / B). Tier C adds ol_intrinsic → descending_neuron.
CUTS = (
    ("ol_intrinsic", "visual_projection"),
    ("visual_projection", "descending_neuron"),
    ("descending_neuron", ("vnc_intrinsic", "vnc_motor")),
)

BUTTONS = ("LF", "RF", "LM", "RM", "LH", "RH")

# θ clamp for Tier B (and as a safety rail on type-gains).
THETA_LO = -2.302585092994046  # log(0.1)
THETA_HI = 2.302585092994046  # log(10)
