"""v7 Figure 9 A-B: effect of adding step 2 in the 12 single-cancer tissue endpoints.

Runs overleaf-tools/make_fig10_v53_two_step_mosaic.py (accepted v6.x Figure 10 grammar) patched in memory: inputs from
the v7 binary-rule outputs (v7_plan/scripts/B6b_binary_rule_external_v7.py), external rows only (the TCGA and other
mixed-cancer series are Figure 9C), legend moved to a row underneath. The original file is not modified.
Output: figures/build/fig9/fig9ab_v7_two_step_tissue.png (stacked with 9C by fig9_v7_assemble.py)
"""
import importlib.util

import v7_style as S

SRC = S.ROOT / "overleaf-tools/make_fig10_v53_two_step_mosaic.py"
BIN = S.RES / "B6_external/two_step_binary"
BUILD = S.V7 / "figures/build/fig9"
REPS = {
    'OUT = ROOT / "overleaf-source-git/figures_300/fig10_v53_two_step_mosaic.png"': f'OUT = Path(r"{BUILD / "fig9ab_v7_two_step_tissue.png"}")',
    'met = pd.read_csv(R / "two_step_binary_rule/binary_rule_metrics.csv")': f'met = pd.read_csv(Path(r"{BIN / "binary_rule_metrics.csv"}"))',
    'cur = pd.read_csv(R / "two_step_binary_rule/binary_rule_decision_curves.csv")': f'cur = pd.read_csv(Path(r"{BIN / "binary_rule_decision_curves.csv"}"))',
    'cmp_ = pd.read_csv(R / "two_step_binary_rule/binary_rule_paired_comparisons.csv")': f'cmp_ = pd.read_csv(Path(r"{BIN / "binary_rule_paired_comparisons.csv"}"))',
    'tc = pd.read_csv(R / "tcga_two_step/results/tcga_assign_confirm_by_assigned_cancer.csv")': "tc = None",
    'tc = tc[tc.threshold == 0.5].copy()': "pass",
    'tc["assigned"] = tc.assigned.replace({"DLBCL": "LYMPH"})': "pass",
    'for lev in ("external", "TCGA"):': 'for lev in ("external",):',
    'ROWS = [("roc", "external"), ("dnb", "external"), (None, None), ("roc", "TCGA"), ("dnb", "TCGA")]':
        'ROWS = [("roc", "external"), ("dnb", "external"), (None, None)]',
    'HR = [1, 1, 0.16, 1, 1]': "HR = [1, 1, 0.55]",
    'ORDER = [c for c in FC.CANCER_ORDER if c not in ("CLL", "MYEL")] + ["CLL", "MYEL"]': "ORDER = list(FC.CANCER_ORDER)",
    'ROC_Y = {"external": ((0.50, 1.025), [0.5, 0.75, 1.0], ["0.5", "0.75", "1"]),':
        'ROC_Y = {"external": ((0.30, 1.025), [0.4, 0.7, 1.0], ["0.4", "0.7", "1"]),',
    'YLIM = {"external": (-25, 36), "TCGA": (-6, 12)}': 'YLIM = {"external": (-45, 40), "TCGA": (-6, 12)}',
    'YTICKS = {"external": [-20, 0, 20], "TCGA": [-5, 0, 5, 10]}': 'YTICKS = {"external": [-40, -20, 0, 20], "TCGA": [-5, 0, 5, 10]}',
    'for rows, lab in [((0, 1), "External"), ((3, 4), "TCGA")]:': 'for rows, lab in [((0, 1), "Tissue")]:',
    'ax_l = fig.add_subplot(gs[3:5, 10:12]); ax_l.axis("off")': 'ax_l = fig.add_subplot(gs[2, 1:11]); ax_l.axis("off")',
    'fontsize=FS["legend"] + 1, handlelength=2.6, handleheight=1.4, labelspacing=1.2,':
        'fontsize=FS["legend"] + 1, handlelength=2.6, handleheight=1.4, labelspacing=1.2, ncol=5, columnspacing=2.5,',
    'loc="center left", frameon=False,\n                fontsize': 'loc="center", frameon=False,\n                fontsize',
}


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    src = SRC.read_text(encoding="utf-8")
    for o, n in REPS.items():
        assert src.count(o) == 1, o
        src = src.replace(o, n)
    spec = importlib.util.spec_from_loader("fig9ab", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(SRC)
    exec(compile(src, "make_fig10_v53 (v7 external rows)", "exec"), mod.__dict__)
    mod.main()


if __name__ == "__main__":
    main()
