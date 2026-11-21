"""v7 Figure 3 panels F-H, revision g (17 Oct: H taller, protein names diagonal; "stack" argument = each sweep
above its heatmap). Revision f (17 Oct: F ring at 26 halved, "26" larger; H top-row heatmaps without
cancer tick labels, rows closer). Revision e (17 Oct: only the selected-size rings enlarged in F and H; H tighter with
the colour-bar label once; F narrower). Revision d (author, 17 Oct): as fig3_v7b_derivation.py, written to build/fig3d, with the
H sweeps-and-heatmaps in 2 rows of 6, bigger sweep markers, and the axes box of F and G recorded for alignment.
Assembled with A-E by fig3_v7d_layout.py.

A-E  v6.x Figure 3 mosaic with the 26-protein list and the v7 disease-specific panels (built by fig3_v7_derivation.py,
     build/fig3/fig3_v7_top.png).
F    v6.x Figure 4A drawing code (fig3_5_and_fig4_5_panel_size_selection.draw_multi_sweep): best candidate per size,
     development 4-fold CV macro AUC, balanced accuracy, macro F1 and minimum recall, band = +/-1 SD across folds
     (was: internal-test values with bootstrap bands); chosen size 26.
G    v6.x Figure 4B (fig4_panels.panel_a via fig3_6_structure_with_sweep.build_partb_panel): the 26 proteins, mean NPX
     z-score by cancer in the development set, bars = |LR-L2 coefficient| from the development candidate bank.
H    v6.x Figure 4C (fig4_7_sweep_structure_and_pairs.build_pairs_panel): per cancer, development 5-fold CV AUC
     (solid, cancer colour) and balanced accuracy (dashed) for the top 1-18 proteins, chosen v7 size marked, next to
     the heatmap of the chosen proteins (development-set mean NPX z-scores).
The figurev6 modules are loaded as patched in-memory copies (sources unchanged; nothing written to figurev6/output or
overleaf-source-git). Output: figures/output/fig3_v7_derivation.png
"""
import importlib.util
import sys
import types

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import v7_style as S

SC = S.ROOT / "figurev6/scripts"
STACK = True                          # stacked H is the main version (author, 17 Oct)
BUILD = S.V7 / ("figures/build/fig3h")
BUILD.mkdir(parents=True, exist_ok=True)
Image.MAX_IMAGE_PIXELS = None

raw = pd.read_csv(S.ROOT / "data/processed/filtered_pancancer_data.csv", usecols=["Sample_ID", "Cancer"])
tr, _ = train_test_split(np.arange(len(raw)), test_size=0.30, random_state=52, stratify=raw.Cancer)
DEV_IDS = set(raw.Sample_ID.iloc[tr])
assert len(DEV_IDS) == 962
pan = pd.read_csv(S.V7 / "panels/panel_memberships_v7.csv")
MC26 = list(pan[pan.panel == "multiclass_26"].protein)
DEPLOYED = {c: int((pan.panel == f"single_{'DLBCL' if c == 'LYMPH' else c}").sum()) for c in S.FC.CANCER_ORDER}


def patched(name, reps, extra=None):
    src = (SC / f"{name}.py").read_text(encoding="utf-8")
    for o, n in reps.items():
        assert src.count(o) >= 1, (name, o)
        src = src.replace(o, n)
    mod = types.ModuleType(name)
    mod.__file__ = str(SC / f"{name}.py")
    mod.__dict__.update(extra or {})
    sys.modules[name] = mod
    exec(compile(src, f"{name} (v7 patched)", "exec"), mod.__dict__)
    return mod


def multi_frame():
    c = pd.read_csv(S.ROOT / "revise_plan/part_b_multiclass/v11_seed52_lr_l2_wide_compact_sensitivity/tables/stage1_cv_results.csv")
    b = c.loc[c.groupby("feature_count").stage1_objective.idxmax()].sort_values("feature_count").reset_index(drop=True)
    out = pd.DataFrame({"feature_count": b.feature_count})
    for m, cv in (("macro_ovr_auc", "cv_macro_ovr_auc"), ("balanced_accuracy", "cv_balanced_accuracy"),
                  ("macro_f1", "cv_macro_f1"), ("min_class_recall", "cv_min_class_recall")):
        out[f"test_{m}"] = b[f"{cv}_mean"]
        out[f"{m}_ci_lower"] = b[f"{cv}_mean"] - b[f"{cv}_std"]
        out[f"{m}_ci_upper"] = b[f"{cv}_mean"] + b[f"{cv}_std"]
    return out


def single_frame(slug):
    sw = pd.read_csv(S.ROOT / "revise_plan/pilot_ds_dev_only_size/size_sweep_devonly.csv")
    d = sw[sw.cancer == slug.upper()].sort_values("size")
    return pd.DataFrame({"Features": d["size"].astype(int).values, "Test_AUC": d.cv_auc.values,
                         "balanced_accuracy": d.cv_balanced_accuracy.values, "f1": np.nan}).reset_index(drop=True)


def soften(C):
    """Author request (16 Oct): lighter, less bold heatmaps. Low end blue-grey and faded, neutral near white, high end
    at a paler purple / cancer colour instead of the deep shade."""
    from matplotlib.colors import LinearSegmentedColormap
    low_dark, low_mid, neutral = "#bcc6d1", "#dde3ea", "#f8f8fa"
    C.PARTB_CMAP = LinearSegmentedColormap.from_list(
        "v7_partb", [(0.00, low_dark), (0.28, low_mid), (0.50, neutral), (0.72, "#d5c3ea"), (1.00, "#9a78c8")], N=256)
    C.cancer_cmap = lambda colour: LinearSegmentedColormap.from_list(
        f"v7_{colour}", [(0.00, low_dark), (0.28, low_mid), (0.50, neutral), (0.72, C._bold(colour, 0.38)),
                         (1.00, C._bold(colour, 0.78))], N=256)


def main():
    import build_slide_mosaics as bsm
    bsm.OUT = BUILD
    import inspect
    bsrc = inspect.getsource(bsm.build).replace('chr(ord("A") + idx)', 'chr(ord(spec.get("first_letter", "A")) + idx)')
    exec(compile(bsrc, "build (first_letter)", "exec"), bsm.__dict__)

    dev_read = {'pd.read_csv(MAIN_DATA, usecols=["Cancer"] + uniq)':
                'pd.read_csv(MAIN_DATA, usecols=["Sample_ID", "Cancer"] + uniq).loc[lambda d: d.Sample_ID.isin(DEV_IDS)]'}
    dev_read = {**dev_read, 'json.dumps({"baseline_frac_from_bottom": float(frac)})':
                'json.dumps({"baseline_frac_from_bottom": float(frac), "box": [(axes_box.x0 - tight.x0 + pad_inches) / (tight.width + 2 * pad_inches), (axes_box.x1 - tight.x0 + pad_inches) / (tight.width + 2 * pad_inches), frac, (axes_box.y1 - tight.y0 + pad_inches) / height]})'}
    C = patched("fig_candidate_common", {**dev_read,
                'OUT = ROOT / "figurev6" / "output"': "OUT = V7_BUILD"} if "OUT = ROOT" in (SC / "fig_candidate_common.py").read_text(encoding="utf-8") else dev_read,
                {"DEV_IDS": DEV_IDS, "V7_BUILD": BUILD})
    soften(C)
    C.PANELS_CSV = S.V7 / "panels/panel_memberships_v7.csv"
    C.RANK_ROOT = S.ROOT / "revise_plan/v5_seed52_common_split/part_a_single"
    M = patched("fig3_5_and_fig4_5_panel_size_selection", {
        'OUT = ROOT / "figurev6" / "output"': "OUT = V7_BUILD",
        'OUT.mkdir(parents=True, exist_ok=True)': "pass",
        'ring(ax, chosen, float(at_chosen[col].iloc[0]), shade, size=6.4)': 'ring(ax, chosen, float(at_chosen[col].iloc[0]), shade, size=14)',
        'ring(ax, LOCKED_SIZE, float(y[x == LOCKED_SIZE].iloc[0]), colour)': 'ring(ax, LOCKED_SIZE, float(y[x == LOCKED_SIZE].iloc[0]), colour, size=12)',
        "LOCKED_SIZE = 25": "LOCKED_SIZE = 26",
        "    ax.set_ylim(-0.03, 1.06)": "    ax.set_ylim(0.48, 1.03)",   # H sweeps: nothing falls below 0.5
        'ax.set_ylabel("Held-out metric value")': 'ax.set_ylabel("Development CV value")',
        'ax.set_ylabel("Held-out value", fontsize=axis_label_size)': 'ax.set_ylabel("Development CV", fontsize=axis_label_size)',
        'label="Selected size (25 markers)"': 'label="Selected size (26 markers)"',
        'label="95% bootstrap interval"': 'label="±1 SD across CV folds"',
        '("Test_AUC", "Held-out AUC",': '("Test_AUC", "CV AUC",',
        '("balanced_accuracy", "Held-out balanced accuracy",': '("balanced_accuracy", "CV balanced accuracy",',
        "    colour = PLOT_COLOR[cancer]\n": "    colour = PLOT_COLOR[cancer]\n    ax.set_facecolor(V7_TINT(colour))\n",
        '    x = df["feature_count"]\n': '    x = df["feature_count"]\n    ax.set_facecolor(V7_TINT("#6a3d9a"))\n',
    }, {"V7_BUILD": BUILD, "V7_TINT": lambda c: S.tint(c, 0.07)})
    M.DEPLOYED = DEPLOYED
    M.SINGLE_SERIES = [s for s in M.SINGLE_SERIES if s[0] != "f1"]   # no CV F1 in the size sweep
    M.load_single = single_frame
    M._load_multi = multi_frame
    F4P = patched("fig4_panels", {'raw      = pd.read_csv(MAIN_DATA, usecols=["Cancer"] + LOCKED_25)':
                                  'raw      = pd.read_csv(MAIN_DATA, usecols=["Sample_ID", "Cancer"] + LOCKED_25).loc[lambda d: d.Sample_ID.isin(DEV_IDS)]',
                                  'OUT    = ROOT / "figurev6" / "output"': "OUT    = V7_BUILD"},
                  {"DEV_IDS": DEV_IDS, "V7_BUILD": BUILD})
    F4P.LOCKED_25 = MC26
    import fig3_6_structure_with_sweep as F36   # binds the patched modules and bsm.OUT = BUILD
    F36.OUT = BUILD
    rot = "\n        for _t in ax_tile.get_yticklabels():\n            _t.set_rotation(35); _t.set_ha('right'); _t.set_rotation_mode('anchor')"
    tile_reps = {"label_size=12.0, tick_size=12.0)": "label_size=11.5, tick_size=12.0, cbar_label=True)\n        if r < NROW - 1:\n            ax_tile.set_xticklabels([])" + rot}
    if STACK:   # sweep above heatmap in each card
        F4 = patched("fig4_7_sweep_structure_and_pairs", {"NROW, NCANC = 3, 4": "NROW, NCANC = 2, 6",
                                                          "figsize=(30.0, 17.0)": "figsize=(38.0, 16.5)", "top=0.97, bottom=0.10)": "top=0.97, bottom=0.15)",
                                                          "hspace=0.52, wspace=0.26": "hspace=0.22, wspace=0.30",
                                                          "subgridspec(1, 2, width_ratios=[1.0, 1.10],": "subgridspec(2, 1, height_ratios=[0.62, 1.20],",
                                                          "wspace=0.46)": "hspace=0.70)",
                                                          "ax_tile = fig.add_subplot(cell[0, 1])": "ax_tile = fig.add_subplot(cell[1, 0])",
                                                          "show_ylabel=(c == 0), show_xlabel=(r == NROW - 1),": "show_ylabel=(c == 0), show_xlabel=False,",
                                                          **tile_reps}, {})
    else:
        F4 = patched("fig4_7_sweep_structure_and_pairs", {"NROW, NCANC = 3, 4": "NROW, NCANC = 2, 6",
                                                          "figsize=(30.0, 17.0)": "figsize=(34.0, 15.5)",
                                                          "hspace=0.52, wspace=0.26": "hspace=0.30, wspace=0.12", "wspace=0.46)": "wspace=0.50)",
                                                          **tile_reps}, {})
    F4.OUT = BUILD

    # F: multiclass sweep (standalone cell as in fig3_5 build_fig3_5, second figure)
    plt = M.plt
    df = multi_frame()
    fig2, ax = plt.subplots(figsize=(7.6, 7.8))
    M.draw_multi_sweep(ax, df)
    fig2.canvas.draw()
    for t in ax.get_xticklabels():
        if t.get_text() == "26":
            t.set_fontsize(t.get_fontsize() * 1.9)
    fig2.legend(handles=M.multi_legend_handles(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01),
                handlelength=2.2, columnspacing=1.4, fontsize=10)
    fig2.tight_layout(rect=(0, 0.135, 1, 1))
    C.save_measured(fig2, BUILD / "f_sweep.png", ax, dpi=400)
    plt.close(fig2)
    # G: heatmap of the 26
    F36.build_partb_panel()
    C.align_images([BUILD / "f_sweep.png", BUILD / f"{F36.PARTB}.png"], [BUILD / "f_sweep_aligned.png", BUILD / "g_heat_aligned.png"])
    # H: per-cancer sweeps + heatmaps
    F4.build_pairs_panel()
    print("wrote F, G, H panels to", BUILD)

if __name__ == "__main__":
    main()
