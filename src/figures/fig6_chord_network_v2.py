#!/usr/bin/env python3
"""
Fig 6 — Protein-pathway circular chord network (full 360°, mosaic-ready)
Proteins on top arc (10°–170°), pathway terms on bottom arc (190°–350°).
DB colors: GO=#7b3f9e  KEGG=#c97b00  Reactome=#1a8b72  (consistent across all fig6 panels).
Protein-function colors deliberately different from all three DB colors.
"""
from __future__ import annotations
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path("e:/Proteomics")
EV   = ROOT / "revise_plan" / "locked25_final_results_package" / "pathway_outputs"
OUT  = ROOT / "figurev5" / "output"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.linewidth": 1.2})
DPI = 300

# ─── Consistent DB colors (matches fig6_panels.py and fig6_parta_opt3) ────────
DB_COL  = {"GO": "#7b3f9e", "KEGG": "#c97b00", "Reactome": "#1a8b72"}
DB_DARK = {"GO": "#4a1870", "KEGG": "#7a4a00", "Reactome": "#0e5040"}

# ─── Protein function colors — chosen to contrast with all DB colors ───────────
# Avoids: purple (#7b3f9e), amber/orange (#c97b00), teal/green (#1a8b72)
PROT_CAT = {
    "FLT3":    "Receptor/kinase",   "FCER2":    "Immune receptor",
    "SLAMF7":  "Immune receptor",   "CXCL17":   "Cytokine",
    "CXCL13":  "Cytokine",          "BMP4":     "Growth factor",
    "PSPN":    "Growth factor",     "TRAF2":    "Signaling",
    "TCL1A":   "Signaling",         "CNTN1":    "Neural adhesion",
    "GFAP":    "Structural",        "ADAMTS13": "Metalloprotease",
    "ADAMTS15":"Metalloprotease",   "CCDC80":   "ECM",
    "LTA4H":   "Enzyme",            "NEFL":     "Structural",
}
CAT_COL = {
    "Receptor/kinase":  "#c62828",  # deep crimson
    "Immune receptor":  "#0d47a1",  # dark navy
    "Cytokine":         "#880e4f",  # maroon-pink
    "Growth factor":    "#2e7d32",  # forest green
    "Signaling":        "#e64a19",  # burnt orange-red
    "Neural adhesion":  "#1565c0",  # royal blue
    "Structural":       "#5d4037",  # dark brown
    "Metalloprotease":  "#c2185b",  # deep pink
    "ECM":              "#546e7a",  # blue-grey
    "Enzyme":           "#558b2f",  # olive green
}

ENTREZ_SYM = {
    "2322": "FLT3",   "2208": "FCER2",  "284340": "CXCL17",
    "10563": "CXCL13","652":  "BMP4",   "5623":   "PSPN",
    "7186":  "TRAF2", "8115": "TCL1A",  "1272":   "CNTN1",
    "57823": "SLAMF7",
}


def _syms(s, db="GO"):
    raw = str(s)
    if db == "GO":
        return [x.strip() for x in raw.split(",") if x.strip()]
    elif db == "KEGG":
        return [ENTREZ_SYM.get(x.strip(), x.strip()) for x in raw.split("/")]
    return [x.strip() for x in raw.split("/")]


def bezier_curve(ax, x0, y0, x1, y1, col, lw, alpha):
    ctrl = 0.22
    verts = [(x0, y0), (x0*ctrl, y0*ctrl), (x1*ctrl, y1*ctrl), (x1, y1)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    ax.add_patch(mpatches.PathPatch(MPath(verts, codes),
                 facecolor="none", edgecolor=col, lw=lw, alpha=alpha, zorder=1))


def arc_segment(ax, theta1, theta2, R, col, lw=9):
    t = np.linspace(theta1, theta2, 80)
    ax.plot(R * np.cos(t), R * np.sin(t),
            color=col, lw=lw, solid_capstyle="round", zorder=3)


def label_node(ax, theta, R_text, label, color, fontsize=11, bold=False):
    x, y = R_text * np.cos(theta), R_text * np.sin(theta)
    deg = np.degrees(theta) % 360
    rot, ha = (deg, "left") if np.cos(theta) >= 0 else (deg - 180, "right")
    ax.text(x, y, label, ha=ha, va="center", fontsize=fontsize, color=color,
            fontweight="bold" if bold else "normal",
            rotation=rot, rotation_mode="anchor", clip_on=False)


# ─── Load data ────────────────────────────────────────────────────────────────
CORR = ROOT/"figurev5"/"output"/"_chord_inputs_corrected"
go_raw    = pd.read_csv(CORR/"go.csv")
kegg_raw  = pd.read_csv(CORR/"kegg.csv")
react_raw = pd.read_csv(CORR/"reactome.csv")

KEGG_SKIP  = {"hsa05169","hsa04913","hsa05320","hsa00620","hsa00480"}
REACT_SKIP = ["nephric duct","Kidney development","B3GALTL","Androgen",
              "Thyroxine","Glycoprotein hormones",
              "Diseases associated with O-glycosylation","O-linked glycosylation"]

kegg_df  = kegg_raw.reset_index(drop=True)
mask     = react_raw["Description"].apply(
    lambda d: not any(kw.lower() in str(d).lower() for kw in REACT_SKIP))
react_df = react_raw.reset_index(drop=True)

rows = []
for _, r in go_raw.iterrows():
    rows.append({"db":"GO",       "term":r["Description"], "padj":r["p.adjust"],
                 "proteins":_syms(r["marker_symbols"], "GO")})
for _, r in kegg_df.iterrows():
    rows.append({"db":"KEGG",     "term":r["Description"], "padj":r["p.adjust"],
                 "proteins":_syms(r["geneID"], "KEGG")})
for _, r in react_df.iterrows():
    rows.append({"db":"Reactome", "term":r["Description"], "padj":r["p.adjust"],
                 "proteins":_syms(r["geneID"], "Reactome")})

terms_df    = pd.DataFrame(rows)
all_prots   = sorted({p for ps in terms_df["proteins"] for p in ps})
prot_sorted = sorted(all_prots, key=lambda p: (PROT_CAT.get(p, "zzz"), p))

terms_sorted = (terms_df
                .assign(_o=terms_df["db"].map({"GO":0,"KEGG":1,"Reactome":2}))
                .sort_values(["_o","padj"]).reset_index(drop=True))

n_prots, n_terms = len(prot_sorted), len(terms_sorted)

# ─── Layout ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 14))
ax.set_aspect("equal")
ax.set_xlim(-1.82, 1.82)
ax.set_ylim(-1.82, 1.82)
ax.axis("off")

R, R_arc, R_dot = 1.00, 1.08, 1.14
R_lab_p, R_lab_t = 1.23, 1.31

prot_theta = np.linspace(np.radians(10),  np.radians(170), n_prots)
term_theta = np.linspace(np.radians(190), np.radians(350), n_terms)

# Guide circle
theta_full = np.linspace(0, 2*np.pi, 400)
ax.plot(R_arc*np.cos(theta_full), R_arc*np.sin(theta_full),
        color="#d8d8d8", lw=1.0, zorder=0)

# ── Protein arcs + dots ───────────────────────────────────────────────────────
hs = np.radians(5.0)
for i, (p, th) in enumerate(zip(prot_sorted, prot_theta)):
    cat = PROT_CAT.get(p, "Other")
    col = CAT_COL.get(cat, "#aaa")
    arc_segment(ax, th-hs, th+hs, R_arc, col, lw=9)
    ax.plot(R_dot*np.cos(th), R_dot*np.sin(th), "o", ms=12,
            color=col, mec="white", mew=1.5, zorder=5)

# ── DB arc spans + term dots ──────────────────────────────────────────────────
db_bounds = {}
for db in ["GO","KEGG","Reactome"]:
    idx = [i for i,r in terms_sorted.iterrows() if r["db"]==db]
    if idx:
        db_bounds[db] = (term_theta[idx[0]], term_theta[idx[-1]])

for db,(t1,t2) in db_bounds.items():
    arc_segment(ax, min(t1,t2)-np.radians(3.5), max(t1,t2)+np.radians(3.5),
                R_arc, DB_COL[db], lw=9)

for i,(_, row) in enumerate(terms_sorted.iterrows()):
    ax.plot(R_dot*np.cos(term_theta[i]), R_dot*np.sin(term_theta[i]),
            "s", ms=11, color=DB_COL[row["db"]], mec="white", mew=1.5, zorder=5)

# ── Bezier edges ─────────────────────────────────────────────────────────────
prot_idx = {p:i for i,p in enumerate(prot_sorted)}

for t_i,(_, trow) in enumerate(terms_sorted.iterrows()):
    th_t = term_theta[t_i]
    tx, ty = R*np.cos(th_t), R*np.sin(th_t)
    db_col = DB_COL[trow["db"]]
    ne = len(trow["proteins"])
    for prot in trow["proteins"]:
        if prot in prot_idx:
            th_p = prot_theta[prot_idx[prot]]
            bezier_curve(ax, R*np.cos(th_p), R*np.sin(th_p), tx, ty,
                         col=db_col, lw=2.2,
                         alpha=0.55 if ne <= 2 else 0.38)

# ── Labels ────────────────────────────────────────────────────────────────────
for i, p in enumerate(prot_sorted):
    col = CAT_COL.get(PROT_CAT.get(p,"Other"), "#333")
    label_node(ax, prot_theta[i], R_lab_p, p, color=col, fontsize=11, bold=True)

for i,(_, trow) in enumerate(terms_sorted.iterrows()):
    label = "\n".join(textwrap.wrap(trow["term"], 22))
    label_node(ax, term_theta[i], R_lab_t, label,
               color=DB_DARK[trow["db"]], fontsize=9.5, bold=True)

# ── Section headers ───────────────────────────────────────────────────────────
ax.text(0,  1.74, "PROTEINS",      ha="center", va="center",
        fontsize=17, fontweight="bold", color="#333333")
ax.text(0, -1.74, "PATHWAY TERMS", ha="center", va="center",
        fontsize=17, fontweight="bold", color="#333333")

# ── Legends ───────────────────────────────────────────────────────────────────
cat_seen = sorted({PROT_CAT.get(p,"Other") for p in prot_sorted})
l1 = ax.legend(
    handles=[mpatches.Patch(facecolor=CAT_COL.get(c,"#aaa"), edgecolor="white", label=c)
             for c in cat_seen],
    title="Protein function", fontsize=10, title_fontsize=11,
    loc="lower left", bbox_to_anchor=(0.01, 0.01), bbox_transform=ax.transAxes,
    framealpha=0.95, edgecolor="#cccccc", ncol=2)
l1.get_title().set_fontweight("bold")
ax.add_artist(l1)

l2 = ax.legend(
    handles=[mpatches.Patch(facecolor=DB_COL[db], edgecolor="white",
                            label=db+" (nominal)")
             for db in ["GO","KEGG","Reactome"]],
    title="Database", fontsize=10, title_fontsize=11,
    loc="lower right", bbox_to_anchor=(0.99, 0.01), bbox_transform=ax.transAxes,
    framealpha=0.95, edgecolor="#cccccc")
l2.get_title().set_fontweight("bold")
ax.add_artist(l2)

ax.legend(
    handles=[plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
                        markersize=12, label="Protein"),
             plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="#888",
                        markersize=11, label="Pathway term")],
    fontsize=10, loc="upper center",
    bbox_to_anchor=(0.5, 0.99), bbox_transform=ax.transAxes,
    ncol=2, framealpha=0.95, edgecolor="#cccccc")

ax.set_title(
    "Protein–pathway circular network — locked 25-protein panel\n"
    "Circle = protein (color = function)   Square = annotated term (color = database)",
    fontsize=13, fontweight="bold", pad=8)

for ext in ("pdf","png"):
    p = OUT / f"fig6_chord_network.{ext}"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    print(f"  saved -> {p}")
plt.close(fig)
print("Done.")
