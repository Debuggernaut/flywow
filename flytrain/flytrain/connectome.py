"""Load MaleCNS feathers the same way fly.py does."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .constants import FAST_NT, GAIN, MIN_WEIGHT, VISUAL_SUPERCLASSES, WSYN


def nt_sign(row: pd.Series) -> int:
    c = row.get("consensus_nt")
    if c in FAST_NT:
        return FAST_NT[c]
    conf = row.get("predicted_nt_confidence")
    pred = row.get("predicted_nt")
    if pd.notna(conf) and conf >= 0.5 and pred in FAST_NT:
        return FAST_NT[pred]
    return FAST_NT.get(row.get("celltype_predicted_nt"), 0)


def load_cells(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    ann = pd.read_feather(data_dir / "body-annotations.feather").rename(
        columns={"bodyId": "body"}
    )
    nt = pd.read_feather(data_dir / "body-neurotransmitters.feather")
    keep_nt = [
        c
        for c in [
            "body",
            "consensus_nt",
            "predicted_nt",
            "predicted_nt_confidence",
            "celltype_predicted_nt",
            "celltype_predicted_nt_confidence",
        ]
        if c in nt.columns
    ]
    if "body" not in nt.columns and "bodyId" in nt.columns:
        nt = nt.rename(columns={"bodyId": "body"})
        keep_nt = ["body"] + [c for c in keep_nt if c != "body"]
    cells = ann.merge(nt[keep_nt], on="body", how="left")
    cells["sign"] = cells.apply(nt_sign, axis=1)
    cells = cells.dropna(subset=["body"]).drop_duplicates("body")
    return cells.reset_index(drop=True)


def load_edges(data_dir: Path, legal_bodies: set[int]) -> pd.DataFrame:
    data_dir = Path(data_dir)
    w = pd.read_feather(data_dir / "connectome-weights.feather").rename(
        columns={"body_pre": "pre", "body_post": "post"}
    )
    return w[
        (w["weight"] >= MIN_WEIGHT)
        & w["pre"].isin(legal_bodies)
        & w["post"].isin(legal_bodies)
        & (w["pre"] != w["post"])
    ].copy()


def build_W(nodes: pd.DataFrame, edges: pd.DataFrame) -> sparse.csr_matrix:
    """W[post, pre] = synapse_count * sign(pre). CSR for matvec."""
    idx = {int(b): i for i, b in enumerate(nodes["body"])}
    pre_i = edges["pre"].map(idx).to_numpy()
    post_i = edges["post"].map(idx).to_numpy()
    sign = nodes.set_index("body")["sign"].reindex(edges["pre"]).fillna(0).to_numpy()
    val = edges["weight"].to_numpy(dtype=np.float32) * sign.astype(np.float32)
    n = len(nodes)
    return sparse.csr_matrix((val, (post_i, pre_i)), shape=(n, n), dtype=np.float32)


def hex_visual_mask(nodes: pd.DataFrame) -> np.ndarray:
    h1 = pd.to_numeric(nodes.get("assignedOlHex1"), errors="coerce")
    has_hex = h1.notna()
    sc = nodes["superclass"].astype(str)
    visual_sc = sc.isin(list(VISUAL_SUPERCLASSES))
    return (has_hex & visual_sc).to_numpy()


def leg_mask(
    nodes: pd.DataFrame, sub: str, side: str, nerve: str | None = None
) -> np.ndarray:
    m = (
        nodes["superclass"].eq("vnc_motor")
        & nodes["subclass"].eq(sub)
        & nodes["somaSide"].eq(side)
    )
    if nerve:
        m = m & nodes["exitNerve"].eq(nerve)
    return m.to_numpy()


def pool_masks(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "LF": leg_mask(nodes, "fl", "L"),
        "RF": leg_mask(nodes, "fl", "R"),
        "LM": leg_mask(nodes, "ml", "L"),
        "RM": leg_mask(nodes, "ml", "R"),
        "LH": leg_mask(nodes, "hl", "L", "MetaLN"),
        "RH": leg_mask(nodes, "hl", "R", "MetaLN"),
    }


def kc_mask(nodes: pd.DataFrame) -> np.ndarray:
    m = nodes["class"].astype(str).eq("Kenyon_Cell")
    if int(m.sum()) == 0:
        m = nodes["type"].astype(str).str.startswith("KC")
    return m.to_numpy()


def sc_eq(nodes: pd.DataFrame, name: str | Iterable[str]) -> np.ndarray:
    sc = nodes["superclass"].astype(str)
    if isinstance(name, str):
        return sc.eq(name).to_numpy()
    return sc.isin(list(name)).to_numpy()


@dataclass
class Connectome:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    W: sparse.csr_matrix
    hex_i: np.ndarray
    vpn_i: np.ndarray
    dn_i: np.ndarray
    vnc_i: np.ndarray
    mn_i: np.ndarray
    kc_i: np.ndarray
    pools: dict[str, np.ndarray]
    pool_i: dict[str, np.ndarray] = field(default_factory=dict)
    hex_px: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hex_py: np.ndarray = field(default_factory=lambda: np.zeros(0))
    types: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    superclasses: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def scale(self) -> float:
        return float(WSYN * GAIN)


def attach_hex_pixels(net: Connectome, canvas: int = 256) -> None:
    from .vision import hex_pixel_xy

    sub = net.nodes.iloc[net.hex_i]
    h1 = pd.to_numeric(sub.get("assignedOlHex1"), errors="coerce").to_numpy()
    h2 = pd.to_numeric(sub.get("assignedOlHex2"), errors="coerce").to_numpy()
    px, py = hex_pixel_xy(h1, h2, canvas)
    net.hex_px = px
    net.hex_py = py


def load_connectome(data_dir: Path) -> Connectome:
    data_dir = Path(data_dir)
    nodes = load_cells(data_dir)
    edges = load_edges(data_dir, set(nodes["body"].astype(int)))
    W = build_W(nodes, edges)
    hex_i = np.flatnonzero(hex_visual_mask(nodes))
    vpn_i = np.flatnonzero(nodes["superclass"].eq("visual_projection"))
    dn_i = np.flatnonzero(nodes["superclass"].eq("descending_neuron"))
    vnc_i = np.flatnonzero(
        nodes["superclass"].isin(["vnc_intrinsic", "vnc_motor"])
    )
    is_mn = nodes["superclass"].isin(["vnc_motor", "cb_motor"]).to_numpy()
    mn_i = np.flatnonzero(is_mn)
    pools = pool_masks(nodes)
    net = Connectome(
        nodes=nodes,
        edges=edges,
        W=W,
        hex_i=hex_i,
        vpn_i=vpn_i,
        dn_i=dn_i,
        vnc_i=vnc_i,
        mn_i=mn_i,
        kc_i=np.flatnonzero(kc_mask(nodes)),
        pools=pools,
        pool_i={k: np.flatnonzero(m) for k, m in pools.items()},
        types=nodes["type"].astype(str).to_numpy(),
        superclasses=nodes["superclass"].astype(str).to_numpy(),
    )
    attach_hex_pixels(net)
    return net


def summarize(net: Connectome) -> dict:
    pools = {k: int(v.sum()) for k, v in net.pools.items()}
    return {
        "n_neurons": int(net.n),
        "n_edges": int(net.W.nnz),
        "n_hex_driven": int(len(net.hex_i)),
        "n_vpn": int(len(net.vpn_i)),
        "n_dn": int(len(net.dn_i)),
        "n_vnc": int(len(net.vnc_i)),
        "n_mn": int(len(net.mn_i)),
        "n_kc": int(len(net.kc_i)),
        "pools": pools,
        "hex_by_superclass": {
            name: int(
                (
                    (net.superclasses == name)
                    & hex_visual_mask(net.nodes)
                ).sum()
            )
            for name in VISUAL_SUPERCLASSES
        },
    }


def load_hex_sampler(data_dir: Path, canvas: int = 256):
    """Hex drive coordinates only — no edges, no LIF. Used by the labeler."""
    from .vision import hex_pixel_xy

    data_dir = Path(data_dir)
    ann_path = data_dir / "body-annotations.feather"
    if not ann_path.exists():
        return None
    ann = pd.read_feather(ann_path).rename(columns={"bodyId": "body"})
    mask = hex_visual_mask(ann)
    sub = ann.loc[mask]
    if len(sub) == 0:
        return None
    h1 = pd.to_numeric(sub.get("assignedOlHex1"), errors="coerce").to_numpy()
    h2 = pd.to_numeric(sub.get("assignedOlHex2"), errors="coerce").to_numpy()
    px, py = hex_pixel_xy(h1, h2, canvas)
    types = sub["type"].astype(str).to_numpy() if "type" in sub.columns else None
    sc = sub["superclass"].astype(str).to_numpy() if "superclass" in sub.columns else None
    return {"px": px, "py": py, "n": int(len(sub)), "types": types, "superclasses": sc}
