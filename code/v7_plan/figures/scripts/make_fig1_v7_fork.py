"""Writes fig1_v7_methods.py: a fork of overleaf-tools/make_fig1_v52_methods.py (original unchanged) with v7 wording
and v7 panel sizes. Run once; then run fig1_v7_methods.py."""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[3]))
from pathlib import Path

BS = chr(92)
src = (PROJECT_ROOT / "overleaf-tools/make_fig1_v52_methods.py").read_text(encoding="utf-8")
reps = {
    '"Disease-specific": (469, 325, ["LASSO ranking", "Test-set size choice", "Final logistic model"], 10.5),':
        '"Disease-specific": (469, 325, ["LASSO ranking", "Development CV size", "Final logistic model"], 10.5),',
    f'"Multi-Classification{BS}nPanels assembly": (751, 439, ["3,000 panels", "CV shortlist: 540", "Test-set choice"], 11),':
        f'"Multi-Classification{BS}nPanels assembly": (751, 439, ["3,000 panels", "Development CV", "Best objective"], 11),',
    '"Multi classification panel": (458, 130, ["25-protein multiclass panel"], 11),':
        '"Multi classification panel": (458, 130, ["26-protein multiclass panel"], 11),',
    '"External transferability protein": (712, 28, ["External marker-set refits", "Protein and transcript data"], 11),':
        '"External transferability protein": (712, 28, ["External marker-set refits", "Tissue and mixed-cancer series"], 11),',
    '''    with (ROOT / "overleaf-source-git/supplement/v5.1/Table_S5_disease_specific_panel_membership.csv").open(encoding="utf-8-sig", newline="") as fh:
        sizes = {row["cancer_code"]: int(row["panel_size"]) for row in csv.DictReader(fh)}''':
        '''    with (ROOT / "v7_plan/panels/panel_memberships_v7.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows_ = list(csv.DictReader(fh))
    sizes = {}
    for row in rows_:
        if row["panel"].startswith("single_"):
            sizes[row["panel"][7:]] = sizes.get(row["panel"][7:], 0) + 1''',
    '    out = src.with_name("fig1_v52_methods.svg")':
        '    out = ROOT / "v7_plan/figures/build/fig1/fig1_v7_methods.svg"\n    out.parent.mkdir(parents=True, exist_ok=True)',
}
for o, n in reps.items():
    assert src.count(o) == 1, o[:70]
    src = src.replace(o, n)
src = src.replace("ROOT = Path(__file__).resolve().parents[1]", 'ROOT = PROJECT_ROOT')
old_w = '    tree.write(out, encoding="utf-8", xml_declaration=True)'
assert src.count(old_w) == 1
src = src.replace(old_w, old_w + chr(10) + '    out.write_text(out.read_text(encoding="utf-8").replace("25-protein panel", "26-protein panel"), encoding="utf-8")')
hdr = ('"""v7 Figure 1: fork of overleaf-tools/make_fig1_v52_methods.py (original unchanged), written by make_fig1_v7_fork.py.\n'
       'Development-only wording (size choice and multiclass choice by development cross-validation), 26-protein panel,\n'
       'v7 disease-specific panel sizes, external box names the tissue and mixed-cancer series. Rendered with headless\n'
       'Chrome as before. Output: figures/build/fig1/fig1_v7_methods.png"""\n')
body = src.split('"""', 2)[2].lstrip("\n")
Path(__file__).with_name("fig1_v7_methods.py").write_text(hdr + body, encoding="utf-8")
print("wrote fig1_v7_methods.py")
