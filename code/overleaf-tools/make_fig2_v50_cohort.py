"""Manuscript v5.0 Figure 2 (cohort and data distribution), new layout.

Supervisor comments on Figure 2: make the y-axis of B match A, and make the
panels, especially D, easier to read. Changes (originals unchanged):
  A  re-rendered by executing figurev6/scripts/fig2a_cohort.py with three patches:
     cancers in the fixed order used by panel B (AML at the top ... PRC at the
     bottom) instead of by group size; legend labels "Development" and "Test"
     (the manuscript's terms); output to the v5 build folder.
  B, C, D  existing figurev6/output renders, unchanged.
  Layout: row 1 = A | B; row 2 = C across the full width; row 3 = D across
  the full width (was a 2 x 2 grid).
Output: overleaf-source-git/figures_300/fig2_v50_cohort_mosaic.png (new file);
figures_300/fig2_slide2_cohort_mosaic.png and figurev6/scripts are untouched.
"""
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "figurev6/scripts"
FIG6OUT = ROOT / "figurev6/output"
BUILD = ROOT / "revise_plan/v5_seed52_common_split/figure_inputs/fig2_build"
LIVE = ROOT / "overleaf-source-git/figures_300/fig2_v50_cohort_mosaic.png"
ORDER = ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC", "GLIOM", "LUNGC", "LYMPH", "MYEL", "OVC", "PRC"]
sys.path.insert(0, str(SCRIPTS))


def render_panel_a() -> None:
    src = (SCRIPTS / "fig2a_cohort.py").read_text(encoding="utf-8")
    reps = {
        'df = df.sort_values("internal_total_n", ascending=True).reset_index(drop=True)':
            'df = df.set_index("cancer").loc[V5_ORDER[::-1]].reset_index()  # barh draws bottom-up',
        'OUT  = ROOT / "figurev6" / "output"': 'OUT  = V5_BUILD',
        'label="Train")': 'label="Development")',
        'label="Held-out test")': 'label="Test")',
    }
    for old, new in reps.items():
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = {"__file__": str(SCRIPTS / "fig2a_cohort.py"), "__name__": "fig2a_v50",
          "V5_ORDER": ORDER, "V5_BUILD": BUILD}
    exec(compile(src, "fig2a_cohort.py (v50 patched)", "exec"), ns)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    render_panel_a()
    import build_slide_mosaics as bsm
    bsm.OUT = BUILD
    base = bsm.SLIDES["fig2_slide2_cohort_mosaic"]
    spec = dict(
        title=base["title"],
        grid=(3, 2),
        source_paths={
            "fig2a_cohort": BUILD / "fig2a_cohort.png",
            "fig2b_npx_heatmap": FIG6OUT / "fig2b_npx_heatmap.png",
            "fig2b_pooled_distributions": FIG6OUT / "fig2b_pooled_distributions.png",
            "fig2f_umap": FIG6OUT / "fig2f_umap.png",
        },
        col_weights=[0.94, 1.06],
        label_band=base.get("label_band", 104),
        panels=[
            ("fig2a_cohort", 0, 0, 1, 1),                 # A
            ("fig2b_npx_heatmap", 0, 1, 1, 1),            # B
            ("fig2b_pooled_distributions", 1, 0, 1, 2),   # C, full width
            ("fig2f_umap", 2, 0, 1, 2),                   # D, full width
        ],
    )
    for name, path in spec["source_paths"].items():
        assert Path(path).exists(), path
    stem = "fig2_v50_cohort_mosaic"
    bsm.build(stem, spec)
    im = Image.open(BUILD / f"{stem}.png")
    width = Image.open(ROOT / "overleaf-source-git/figures_300/fig2_slide2_cohort_mosaic.png").size[0]
    im.resize((width, round(im.height * width / im.width)), Image.LANCZOS).save(LIVE, optimize=True)
    print(LIVE, Image.open(LIVE).size)


if __name__ == "__main__":
    main()
