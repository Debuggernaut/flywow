"""Inventory and extract the three (plus optional Tier C) trainable cuts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .connectome import Connectome


def _as_set(name) -> set[str]:
    if isinstance(name, (tuple, list, set)):
        return set(name)
    return {name}


def edge_cut_mask(net: Connectome, pre_sc, post_sc) -> np.ndarray:
    pre_ok = net.nodes.set_index("body")["superclass"].astype(str)
    pre = pre_ok.reindex(net.edges["pre"]).isin(_as_set(pre_sc)).to_numpy()
    post = pre_ok.reindex(net.edges["post"]).isin(_as_set(post_sc)).to_numpy()
    return pre & post


def count_cut(net: Connectome, pre_sc, post_sc) -> dict:
    m = edge_cut_mask(net, pre_sc, post_sc)
    sub = net.edges.loc[m]
    pre_types = (
        net.nodes.set_index("body")["type"]
        .astype(str)
        .reindex(sub["pre"])
        .fillna("?")
    )
    post_types = (
        net.nodes.set_index("body")["type"]
        .astype(str)
        .reindex(sub["post"])
        .fillna("?")
    )
    pairs = pd.Series(list(zip(pre_types.to_numpy(), post_types.to_numpy())))
    return {
        "pre": pre_sc if isinstance(pre_sc, str) else list(pre_sc),
        "post": post_sc if isinstance(post_sc, str) else list(post_sc),
        "n_edges": int(m.sum()),
        "n_pre_cells": int(sub["pre"].nunique()) if len(sub) else 0,
        "n_post_cells": int(sub["post"].nunique()) if len(sub) else 0,
        "n_pre_types": int(pre_types.nunique()) if len(sub) else 0,
        "n_post_types": int(post_types.nunique()) if len(sub) else 0,
        "n_type_pairs": int(pairs.nunique()) if len(sub) else 0,
        "weight_sum": float(sub["weight"].sum()) if len(sub) else 0.0,
    }


def inventory(net: Connectome) -> list[dict]:
    specs = [
        ("ol_intrinsic", "visual_projection"),
        ("visual_projection", "descending_neuron"),
        ("descending_neuron", ("vnc_intrinsic", "vnc_motor")),
        ("ol_intrinsic", "descending_neuron"),  # Tier C
        ("descending_neuron", "vnc_motor"),
        ("visual_projection", "vnc_motor"),
        ("ol_intrinsic", "vnc_motor"),
    ]
    # Restrict first cut to hex-driven ol_intrinsic when possible.
    rows = [count_cut(net, a, b) for a, b in specs]
    hex_bodies = set(net.nodes.iloc[net.hex_i]["body"].astype(int))
    m = edge_cut_mask(net, "ol_intrinsic", "visual_projection")
    m_hex = m & net.edges["pre"].isin(hex_bodies).to_numpy()
    rows.append(
        {
            "pre": "ol_intrinsic(hex-driven)",
            "post": "visual_projection",
            "n_edges": int(m_hex.sum()),
            "n_pre_cells": int(net.edges.loc[m_hex, "pre"].nunique()) if m_hex.any() else 0,
            "n_post_cells": int(net.edges.loc[m_hex, "post"].nunique()) if m_hex.any() else 0,
            "n_pre_types": None,
            "n_post_types": None,
            "n_type_pairs": None,
            "weight_sum": float(net.edges.loc[m_hex, "weight"].sum()) if m_hex.any() else 0.0,
        }
    )
    return rows


def missing_path_report(rows: list[dict]) -> list[str]:
    """If a required hop is empty, say so. Do not raise GAIN."""
    alerts = []
    need = {
        ("ol_intrinsic", "visual_projection"),
        ("visual_projection", "descending_neuron"),
        ("descending_neuron", "vnc_motor"),
    }
    by = {(tuple(r["pre"]) if isinstance(r["pre"], list) else r["pre"],
           tuple(r["post"]) if isinstance(r["post"], list) else r["post"]): r for r in rows}
    for a, b in need:
        r = by.get((a, b))
        if r is None:
            continue
        if r["n_edges"] == 0:
            alerts.append(
                f"EMPTY CUT {a} → {b} at weight>=5. Move the trainable cut one hop "
                f"earlier (Tier C). Do not raise global GAIN."
            )
    return alerts


@dataclass
class CutMatrices:
    """Sparse submatrices for the linear multi-hop Stage B model.

    Each cut is stored as CSR with shape (n_post_global, n_pre_global) so we
    can slice with the cached rate vectors aligned to net indices. For speed
    we also keep compact (n_post_subset, n_pre_subset) forms.
    """

    name: str
    pre_i: np.ndarray
    post_i: np.ndarray
    W_compact: sparse.csr_matrix  # (len(post_i), len(pre_i))
    pre_type: np.ndarray
    post_type: np.ndarray
    edge_pre_local: np.ndarray
    edge_post_local: np.ndarray
    edge_w0: np.ndarray
    edge_pre_type: np.ndarray
    edge_post_type: np.ndarray


def extract_cut(net: Connectome, pre_sc, post_sc, name: str) -> CutMatrices:
    pre_mask = net.superclasses == pre_sc if isinstance(pre_sc, str) else np.isin(
        net.superclasses, list(pre_sc)
    )
    post_mask = net.superclasses == post_sc if isinstance(post_sc, str) else np.isin(
        net.superclasses, list(post_sc)
    )
    # For ol_intrinsic cut, keep only hex-driven sources (the ones we stimulate).
    if pre_sc == "ol_intrinsic":
        hex_mask = np.zeros(net.n, dtype=bool)
        hex_mask[net.hex_i] = True
        pre_mask = pre_mask & hex_mask

    pre_i = np.flatnonzero(pre_mask)
    post_i = np.flatnonzero(post_mask)
    pre_map = {int(i): k for k, i in enumerate(pre_i)}
    post_map = {int(i): k for k, i in enumerate(post_i)}

    # Walk CSR columns of the global W restricted to these sets.
    W = net.W.tocsc()
    rows = []
    cols = []
    data = []
    pre_types_all = []
    post_types_all = []
    for j, gi in enumerate(pre_i):
        a, b = W.indptr[gi], W.indptr[gi + 1]
        dest = W.indices[a:b]
        val = W.data[a:b]
        for d, v in zip(dest, val):
            lk = post_map.get(int(d))
            if lk is None:
                continue
            rows.append(lk)
            cols.append(j)
            data.append(float(v))
            pre_types_all.append(net.types[gi])
            post_types_all.append(net.types[d])

    Wc = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(len(post_i), len(pre_i)),
        dtype=np.float32,
    )
    return CutMatrices(
        name=name,
        pre_i=pre_i,
        post_i=post_i,
        W_compact=Wc,
        pre_type=net.types[pre_i],
        post_type=net.types[post_i],
        edge_pre_local=np.asarray(cols, dtype=np.int32),
        edge_post_local=np.asarray(rows, dtype=np.int32),
        edge_w0=np.asarray(data, dtype=np.float32),
        edge_pre_type=np.asarray(pre_types_all, dtype=object),
        edge_post_type=np.asarray(post_types_all, dtype=object),
    )


def extract_standard_cuts(net: Connectome) -> dict[str, CutMatrices]:
    cuts = {
        "ol_vpn": extract_cut(net, "ol_intrinsic", "visual_projection", "ol_vpn"),
        "vpn_dn": extract_cut(net, "visual_projection", "descending_neuron", "vpn_dn"),
        "dn_vnc": extract_cut(
            net, "descending_neuron", ("vnc_intrinsic", "vnc_motor"), "dn_vnc"
        ),
        "ol_dn": extract_cut(net, "ol_intrinsic", "descending_neuron", "ol_dn"),
    }
    return cuts
