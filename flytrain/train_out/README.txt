flytrain output
================

This folder holds type-gains for existing MaleCNS synapses.
Sparsity and signs are frozen. Global GAIN stays 0.2.

LIF constants (must match fly.py)
---------------------------------
TICK_HZ=100
INNER_STEPS=2
INNER_DT=0.005
WINDOW_INNER_STEPS=20
TAU_M=0.02  TAU_S=0.005
VREST=-52.0  VTH=-45.0  VRESET=-52.0  TREF=0.0022
WSYN=0.275  GAIN=0.2  MIN_WEIGHT=5
FOV_PX=256  VISION_HZ_MAX=80.0  FOV_SHIFT_X_FRAC=0.18  FOV_SHIFT_Y_FRAC=0.0

How fly.py should apply gains
-----------------------------
After build_W(...).tocsc():

    from flytrain.bake import apply_type_gains
    apply_type_gains(W, nodes, "gains_type.csv")

apply_type_gains multiplies W.data on matching (type_pre, type_post) edges
in the three allowed cuts by exp(theta). Unlisted types keep gain 1.

Files
-----
gains_type.csv     type_pre, type_post, exp_theta, cut
metrics.json       val BCE, per-button AP, dark/white probes
cache/             Stage A LIF rate vectors (optional)

What this is not
----------------
Not a biophysical fly. Not a gait CPG. Not a pixel classifier.
Synapse `weight` is a count × guessed NT sign × learned scale.
Kenyon cells / mushroom body were not trained.
