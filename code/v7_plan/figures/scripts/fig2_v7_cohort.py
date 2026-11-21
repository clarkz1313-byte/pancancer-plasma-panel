"""v7 Figure 2: cohort and plasma protein overview. Runs overleaf-tools/make_fig2_v51_cohort.py (patched in memory;
original unchanged) with two extra changes to panel D (author request, 16 Oct): the coloured frames around the 12
single-cancer UMAP tiles are removed (titles in the cancer colour identify them) and the titles sit closer to the
points (pad 4 -> 0). Data unchanged. Output: figures/output/fig2_v7_cohort.png
"""
import importlib.util

import v7_style as S

SRC = S.ROOT / "overleaf-tools/make_fig2_v51_cohort.py"
BUILD = S.V7 / "figures/build/fig2"
EXTRA = ('        "ax.set_title(main_label(c), fontsize=9.5,": "ax.set_title(main_label(c), fontsize=21,",\n'
         '        "color=CANCER_COLOR[c], pad=4)": "color=CANCER_COLOR[c], pad=0)",\n'
         '        "        sp.set_edgecolor(CANCER_COLOR[c])\\n        sp.set_linewidth(1.8)": "        sp.set_visible(False)",\n')
REPS = {
    'BUILD = ROOT / "revise_plan/v5_seed52_common_split/figure_inputs/fig2_v51_build"': f'BUILD = Path(r"{BUILD}")',
    'LIVE = ROOT / "overleaf-source-git/figures_300/fig2_v51_cohort_mosaic.png"': f'LIVE = Path(r"{S.OUT / "fig2_v7_cohort.png"}")',
    '        "ax.set_title(main_label(c), fontsize=9.5,": "ax.set_title(main_label(c), fontsize=21,",\n': EXTRA,
}


def main():
    src = SRC.read_text(encoding="utf-8")
    for o, n in REPS.items():
        assert src.count(o) == 1, o
        src = src.replace(o, n)
    BUILD.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_loader("fig2_v7", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(SRC)
    exec(compile(src, "make_fig2_v51_cohort (v7)", "exec"), mod.__dict__)
    mod.main()


if __name__ == "__main__":
    main()
