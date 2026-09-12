#!/usr/bin/env python3
"""Who actually synapses onto Kenyon cells in the local Feather dump?"""

from pathlib import Path
import pandas as pd

DATA = Path(r"C:\Dev\flywow\data")

ann = pd.read_feather(DATA / "body-annotations.feather").rename(columns={"bodyId": "body"})
w = pd.read_feather(DATA / "connectome-weights.feather").rename(
    columns={"body_pre": "pre", "body_post": "post"}
)

kc = ann[ann["class"].astype(str).eq("Kenyon_Cell")]
if len(kc) == 0:
    kc = ann[ann["type"].astype(str).str.startswith("KC")]
kc_ids = set(kc["body"].astype(int))
print(f"KCs {len(kc)}  types:\n{kc['type'].value_counts().head(15).to_string()}")

onto = w[w["post"].isin(kc_ids)]
print(f"\nraw edges onto KCs: {len(onto):,}")
print(onto["weight"].describe().to_string())

onto = onto.merge(ann[["body", "type", "class", "superclass"]], left_on="pre", right_on="body")
print("\nwho talks to KCs, any weight:")
print(
    onto.groupby(["superclass", "class", "type"], dropna=False)["weight"]
    .agg(["size", "sum"])
    .sort_values("sum", ascending=False)
    .head(25)
    .to_string()
)

for min_w in (1, 5, 10):
    e = onto[onto["weight"] >= min_w]
    vis = e[e["superclass"].eq("visual_projection") | e["class"].astype(str).str.contains("visual", case=False, na=False)]
    print(f"\nmin_w={min_w}  edges={len(e):,}  visual-ish sources={vis['pre'].nunique()} syn={vis['weight'].sum()}")

print("\nvisual ol_sensory -> KC (should be ~0; PRs do not hit the calyx):")
pr = set(
    ann[(ann.superclass == "ol_sensory") & (ann["class"].astype(str).eq("visual"))]["body"].astype(int)
)
print(len(onto[onto["pre"].isin(pr)]))
