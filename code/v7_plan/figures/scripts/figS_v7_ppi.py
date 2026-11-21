"""v7 Supplementary Figure: STRING networks (v6.x Figure 10 / figurev6 fig10_5_ppi_combined).

Runs overleaf-tools/make_fig10_v50_ppi.py patched in memory: statistics from the v7 B9 STRING re-audit
(results/B9_biology/string/ppi_reaudit_combined_results_v2.csv), per-panel and pooled STRING exports from
v7_plan/scripts/B9c_string_figures_v7.py, and the multiclass network from B9d_string_multiclass26_v7.py (26 proteins;
the figure code's LOCKED25 file names are kept). The original files are not modified.
Output: figures/output/figS_v7_ppi.png
"""
import importlib.util

import v7_style as S

SRC = S.ROOT / "overleaf-tools/make_fig10_v50_ppi.py"
B9 = S.RES / "B9_biology"
MC = B9 / "string_figures/STRING_analysis_multiclass26/query_only"
REPS = {
    'BUILD = V5 / "figure_inputs/fig10_build"': f'BUILD = Path(r"{S.V7 / "figures/build/figS_ppi"}")',
    'STATS = V5 / "biology/string/ppi_reaudit_combined_results_v2.csv"': f'STATS = Path(r"{B9 / "string/ppi_reaudit_combined_results_v2.csv"}")',
    'SF = V5 / "biology/string_figures"': f'SF = Path(r"{B9 / "string_figures"}")',
    'LIVE = ROOT / "overleaf-source-git/figures_300/fig10_v50_ppi_combined.png"': f'LIVE = Path(r"{S.OUT / "figS_v7_ppi.png"}")',
    "    B.REAUDIT_CSV = C.REAUDIT_CSV = D.REAUDIT_CSV = STATS":
        "    B.REAUDIT_CSV = C.REAUDIT_CSV = D.REAUDIT_CSV = STATS\n"
        f"    B.SRC = Path(r\"{MC}\")\n"
        "    for _a, _f in (('PNG', 'LOCKED25_query_only_official_STRING_v12.png'), ('PPI_TSV', 'ppi_enrichment.tsv'), ('NETWORK', 'network.tsv')):\n"
        "        if hasattr(B, _a):\n"
        "            setattr(B, _a, B.SRC / _f)",
}


def main():
    src = SRC.read_text(encoding="utf-8")
    for o, n in REPS.items():
        assert src.count(o) == 1, o
        src = src.replace(o, n)
    (S.V7 / "figures/build/figS_ppi").mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_loader("figS_ppi", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(SRC)
    exec(compile(src, "make_fig10_v50_ppi (v7 inputs)", "exec"), mod.__dict__)
    mod.main()


if __name__ == "__main__":
    main()
