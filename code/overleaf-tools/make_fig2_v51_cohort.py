"""Manuscript v5.0 Figure 2, layout v51: four full-width rows (A, B, C, D).

Supervisor comment on Figure 2: same y-axis order in A and B, larger panels,
and D "very small and hard to see". v51 (make_fig2_v50_cohort.py and its PNG
are kept as backup; figurev6 scripts and outputs are unchanged):
  A  figurev6/scripts/fig2a_cohort.py executed with its plotting block replaced
     by a column chart: cancers left to right in the heatmap order, stacked
     development and test counts, totals above the bars; wide and short.
  B  figurev6/scripts/fig2b_npx_heatmap.py executed with a wider figure size.
  C  figurev6/scripts/fig2b_pooled_distributions.py executed with a white plot
     background (was #fafafa) and a wider figure size.
  D  figurev6/scripts/fig2f_umap.py executed with the filled 95% ellipses
     removed (twelve overlapping ellipses hid the points), points more opaque,
     white backgrounds, and the twelve single-cancer tiles in two rows of six.
Output: overleaf-source-git/figures_300/fig2_v51_cohort_mosaic.png (new file).
"""
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "figurev6/scripts"
BUILD = ROOT / "revise_plan/v5_seed52_common_split/figure_inputs/fig2_v51_build"
LIVE = ROOT / "overleaf-source-git/figures_300/fig2_v51_cohort_mosaic.png"
ORDER = ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC", "GLIOM", "LUNGC", "LYMPH", "MYEL", "OVC", "PRC"]
sys.path.insert(0, str(SCRIPTS))

PANEL_A = '''fig, ax = plt.subplots(figsize=(18, 3.9))
df = df.set_index("cancer").loc[V5_ORDER].reset_index()
for idx, row in df.iterrows():
    c = row["cancer"]
    col = CANCER_COLOR.get(c, "#888888")
    dev = int(round(row["internal_train_dev_n_estimate"]))
    test = int(row["heldout_test_n"])
    total = int(row["internal_total_n"])
    ax.bar(idx, dev, color=col, width=0.72, zorder=2)
    ax.bar(idx, test, bottom=dev, color=lighter(col), width=0.72, zorder=2)
    if dev >= 25:
        ax.text(idx, dev / 2, str(dev), ha="center", va="center", fontsize=13,
                fontweight="bold", color="white")
    if test >= 25:
        ax.text(idx, dev + test / 2, str(test), ha="center", va="center", fontsize=13,
                fontweight="bold", color=col)
    ax.text(idx, total + 6, str(total), ha="center", va="bottom", fontsize=14,
            fontweight="bold", color=col)
ax.set_xticks(range(len(df)))
ax.set_xticklabels([cancer_label(c) for c in df["cancer"]], fontsize=15, fontweight="bold")
for tick, c in zip(ax.get_xticklabels(), df["cancer"]):
    tick.set_color(CANCER_COLOR.get(c, "#333333"))
ax.set_ylabel("Patients", fontsize=16, fontweight="bold", labelpad=6)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, df["internal_total_n"].max() * 1.18)
ax.set_xlim(-0.6, len(df) - 0.4)
ax.tick_params(axis="y", labelsize=14)
ax.tick_params(axis="x", length=0)
ax.legend(handles=[Patch(facecolor="#555555", label="Development"),
                   Patch(facecolor=lighter("#555555"), label="Test")],
          fontsize=14, framealpha=0.0, loc="upper right", ncol=2)
ax.grid(axis="y", color="#e8e8e8", lw=0.9, zorder=0)
ax.set_axisbelow(True)
'''


def run_patched(script: str, reps: dict, extra: dict | None = None) -> None:
    path = SCRIPTS / script
    src = path.read_text(encoding="utf-8")
    for old, new in reps.items():
        assert src.count(old) == 1, (script, old[:60])
        src = src.replace(old, new)
    ns = {"__file__": str(path), "__name__": "v51_" + path.stem, "V5_BUILD": BUILD}
    ns.update(extra or {})
    exec(compile(src, f"{script} (v51 patched)", "exec"), ns)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    out_line = 'OUT  = ROOT / "figurev6" / "output"'

    src_a = (SCRIPTS / "fig2a_cohort.py").read_text(encoding="utf-8")
    a0 = src_a.index("fig, ax = plt.subplots(figsize=(10, 7))")
    a1 = src_a.index("ax.set_axisbelow(True)\n", a0) + len("ax.set_axisbelow(True)\n")
    run_patched("fig2a_cohort.py", {out_line: "OUT  = V5_BUILD", src_a[a0:a1]: PANEL_A}, {"V5_ORDER": ORDER})

    run_patched("fig2b_npx_heatmap.py", {out_line: "OUT  = V5_BUILD",
                                         "fig, ax = plt.subplots(figsize=(14.0, 7.6))":
                                         "fig, ax = plt.subplots(figsize=(18.0, 5.6))"})

    run_patched("fig2b_pooled_distributions.py", {out_line: "OUT  = V5_BUILD",
                                                  'ax.set_facecolor("#fafafa")': 'ax.set_facecolor("white")',
                                                  "fig, ax = plt.subplots(figsize=(13, 6.2))":
                                                  "fig, ax = plt.subplots(figsize=(18, 5.4))",
                                                  'bbox_to_anchor=(0.5, -0.13),\n          ncol=8,':
                                                  'bbox_to_anchor=(0.5, -0.20),\n          ncol=15,'})

    run_patched("fig2f_umap.py", {
        out_line: "OUT  = V5_BUILD",
        'ax_main.set_facecolor("#fafafa")': 'ax_main.set_facecolor("white")',
        'ax.set_facecolor("#f5f5f5")': 'ax.set_facecolor("white")',
        "c=CANCER_COLOR[c], s=16, alpha=0.28,": "c=CANCER_COLOR[c], s=26, alpha=0.55,",
        # centre labels overlapped in the middle of the embedding: replaced by a colour legend
        '    ax_main.text(embedding[mask, 0].mean(), embedding[mask, 1].mean(), main_label(c),\n'
        '                 fontsize=9, fontweight="bold",\n'
        '                 ha="center", va="center", color="white", zorder=5,\n'
        '                 path_effects=[pe.withStroke(linewidth=3.0, foreground=CANCER_COLOR[c])])':
            "    pass",
        'ax_main.set_xlabel("UMAP-1", fontsize=13': 'ax_main.set_xlabel("UMAP-1", fontsize=20',
        'ax_main.set_ylabel("UMAP-2", fontsize=13': 'ax_main.set_ylabel("UMAP-2", fontsize=20',
        "ax_main.tick_params(labelsize=10)": "ax_main.tick_params(labelsize=16)",
        "ax_main.set_axisbelow(True)":
            "ax_main.set_axisbelow(True)\n"
            "ax_main.legend(handles=[plt.Line2D([0], [0], marker='o', lw=0, color=CANCER_COLOR[c], markersize=12,\n"
            "                                   label=main_label(c)) for c in CANCERS],\n"
            "               ncol=4, fontsize=16, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False,\n"
            "               handletextpad=0.2, columnspacing=0.9, labelspacing=0.35)",
        'c="#c4c4c4", s=5, alpha=0.20,': 'c="#c4c4c4", s=9, alpha=0.25,',
        "c=CANCER_COLOR[c], s=16, alpha=0.82,": "c=CANCER_COLOR[c], s=26, alpha=0.85,",
        "ax.set_title(main_label(c), fontsize=9.5,": "ax.set_title(main_label(c), fontsize=21,",
        "    confidence_ellipse(ax_main, embedding[mask, 0], embedding[mask, 1],\n"
        "                       color=CANCER_COLOR[c], n_std=2.0, alpha_fill=0.13, lw=2.0)\n":
            "    pass  # v51: filled 95% ellipses removed; they overlapped into one dark area\n",
        "fig = plt.figure(figsize=(21, 10))": "fig = plt.figure(figsize=(24, 7.6))",
        "gs  = GridSpec(3, 6, figure=fig,\n               width_ratios=[2.5, 0.04, 1.0, 1.0, 1.0, 1.0])":
            "gs  = GridSpec(2, 8, figure=fig,\n               width_ratios=[2.2, 0.04, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])",
        "    row = i // 4\n    col = (i % 4) + 2       # columns 2-5":
            "    row = i // 6\n    col = (i % 6) + 2       # columns 2-7",
    })

    import build_slide_mosaics as bsm
    bsm.OUT = BUILD
    spec = dict(
        title=bsm.SLIDES["fig2_slide2_cohort_mosaic"]["title"],
        grid=(4, 1),
        label_band=104,
        source_paths={n: BUILD / f"{n}.png" for n in
                      ("fig2a_cohort", "fig2b_npx_heatmap", "fig2b_pooled_distributions", "fig2f_umap")},
        panels=[("fig2a_cohort", 0, 0, 1, 1), ("fig2b_npx_heatmap", 1, 0, 1, 1),
                ("fig2b_pooled_distributions", 2, 0, 1, 1), ("fig2f_umap", 3, 0, 1, 1)],
    )
    stem = "fig2_v51_cohort_mosaic"
    bsm.build(stem, spec)
    im = Image.open(BUILD / f"{stem}.png")
    im.resize((3600, round(im.height * 3600 / im.width)), Image.LANCZOS).save(LIVE, optimize=True, dpi=(300, 300))
    print(LIVE, Image.open(LIVE).size)


if __name__ == "__main__":
    main()
