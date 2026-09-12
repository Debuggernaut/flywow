"""Smoke the trainer on the synthetic graph (no MaleCNS feathers required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flytrain.connectome import summarize
from flytrain.constants import FOV_PX, INNER_DT, WINDOW_INNER_STEPS
from flytrain.cuts import extract_standard_cuts, inventory
from flytrain.dataset import examples_from_index, load_index, split_by_clip
from flytrain.synth import make_synthetic_connectome, write_synthetic_labels
from flytrain.vision import fit_frame_to_fov, hex_pixel_xy, hex_rates_from_fov
from PIL import Image


def test_constants_match_spec():
    assert abs(INNER_DT - 0.005) < 1e-12
    assert WINDOW_INNER_STEPS == 20


def test_hex_pixel_xy_unit():
    h1 = np.array([0.0, 1.0, 0.0, 10.0])
    h2 = np.array([0.0, 0.0, 1.0, 10.0])
    px, py = hex_pixel_xy(h1, h2, 256)
    assert px.shape == h1.shape
    assert np.all((px >= 0) & (px <= 256))
    assert np.all((py >= 0) & (py <= 256))


def test_letterbox():
    im = Image.new("RGB", (3840, 2160), (12, 34, 56))
    out = fit_frame_to_fov(im, FOV_PX)
    assert out.size == (FOV_PX, FOV_PX)
    arr = np.asarray(out)
    # letterbox bars are black
    assert arr[0, 0].tolist() == [0, 0, 0]
    assert arr[FOV_PX // 2, FOV_PX // 2].tolist() == [12, 34, 56]


def test_synth_graph_has_three_cuts():
    net = make_synthetic_connectome()
    info = summarize(net)
    assert info["n_hex_driven"] > 0
    assert info["n_dn"] > 0
    for k, n in info["pools"].items():
        assert n > 0, k
    rows = { (str(r["pre"]), str(r["post"])): r for r in inventory(net) }
    assert rows[("ol_intrinsic", "visual_projection")]["n_edges"] > 0
    assert rows[("visual_projection", "descending_neuron")]["n_edges"] > 0
    cuts = extract_standard_cuts(net)
    assert cuts["ol_vpn"].W_compact.nnz > 0
    assert cuts["vpn_dn"].W_compact.nnz > 0
    assert cuts["dn_vnc"].W_compact.nnz > 0


def test_hex_rates_bright_patch_not_uniform(tmp_path: Path):
    net = make_synthetic_connectome()
    from flytrain.vision import make_patch_fov

    lo = net.hex_px < np.median(net.hex_px)
    hi = ~lo
    left = make_patch_fov(float(net.hex_px[lo].mean()), float(net.hex_py[lo].mean()), radius=24)
    right = make_patch_fov(float(net.hex_px[hi].mean()), float(net.hex_py[hi].mean()), radius=24)
    rL = hex_rates_from_fov(left, net.hex_px, net.hex_py)
    rR = hex_rates_from_fov(right, net.hex_px, net.hex_py)
    assert rL.max() > 0 and rR.max() > 0
    # Different patches should drive a different subset
    assert not np.allclose(rL, rR)


def test_end_to_end_train(tmp_path: Path):
    from flytrain.cache import cache_examples
    from flytrain.train import train_tier_a
    from flytrain.bake import bake_folder, apply_type_gains
    from flytrain.cache import simulate_fov
    from flytrain.vision import make_dark_fov

    net = make_synthetic_connectome()
    labels = tmp_path / "labels"
    write_synthetic_labels(labels, n_per_button=3, seed=1)
    df, root = load_index(labels)
    examples = split_by_clip(examples_from_index(df, root), val_frac=0.25, seed=1)
    cache_dir = tmp_path / "cache"
    cache_examples(net, examples, cache_dir, seed=1)
    from flytrain.cache import load_cache

    cache, meta = load_cache(cache_dir)
    assert cache["y"].shape[1] == 6
    out = tmp_path / "out"
    res = train_tier_a(net, cache, meta, out, steps=80, lr=2e-2, pair=False)
    assert (out / "gains_type.csv").exists()
    bake_folder(out, net)
    W2 = apply_type_gains(net.W, net.nodes, out / "gains_type.csv")
    dark = simulate_fov(net, make_dark_fov(), seed=3, W=W2)
    # Dark should not explode. Synthetic graph is small; just bound it.
    assert float(dark["pool_hz"].sum()) < 50.0
    assert res.metrics["n_params_theta"] > 0
