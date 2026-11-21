"""Manuscript v5.0 Figure 10 (STRING networks; figurev6 fig10_5_ppi_combined).

Re-renders the four lean sub-panels and the combined figure with the unchanged
figurev6 drawing code, pointing module paths at v5 inputs:
  A fig11_panels              IMG_DIR -> v5 per-panel STRING exports (script 56)
  B fig10_panels              REAUDIT_CSV -> v5 statistics (25-protein network
                              unchanged; its 14-test q changes with the v5 family)
  C fig11c_pooled12panels     SRC -> v5 pooled STRING query (script 56); REAUDIT_CSV -> v5
  D fig11d_pooled12panels     same as C
  statistics: revise_plan/v5_seed52_common_split/biology/string/ppi_reaudit_combined_results_v2.csv (script 47)
Nothing in figurev6/output is written; the build goes to
revise_plan/v5_seed52_common_split/figure_inputs/fig10_build/.
Output: overleaf-source-git/figures_300/fig10_v50_ppi_combined.png (new file).
"""
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "revise_plan/v5_seed52_common_split"
BUILD = V5 / "figure_inputs/fig10_build"
STATS = V5 / "biology/string/ppi_reaudit_combined_results_v2.csv"
SF = V5 / "biology/string_figures"
REF = ROOT / "overleaf-source-git/figures_300/fig10_5_ppi_combined.png"
LIVE = ROOT / "overleaf-source-git/figures_300/fig10_v50_ppi_combined.png"
sys.path.insert(0, str(ROOT / "figurev6/scripts"))


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    import fig11_panels as A
    import fig10_panels as B
    import fig11c_pooled12panels_network as C
    import fig11d_pooled12panels_official as D
    import fig10_5_ppi_combined as F

    pooled = SF / "STRING_analysis_pooled12panels"
    for m in (A, B, C, D, F):
        m.OUT = BUILD
    A.load_stats = lambda: None      # loaded but not drawn by fig11_panels.main
    A.load_reaudit = lambda: None
    B.REAUDIT_CSV = C.REAUDIT_CSV = D.REAUDIT_CSV = STATS

    # fig11d reads a label-adjusted render of the pooled SVG. No script in the
    # project produced the v4 one; generate it with fig11_panels' own
    # _readable_string_render (the routine used for panel A).
    tmp = BUILD / "pooled_readable_src" / "POOLED12"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "POOLED12_query_only_official_STRING_v12.svg").write_bytes(
        (pooled / "POOLED12_query_only_official_STRING_v12.svg").read_bytes())
    A.IMG_DIR = tmp.parent
    readable = A._readable_string_render("POOLED12")
    (pooled / "POOLED12_query_only_official_STRING_v12_readable.png").write_bytes(readable.read_bytes())

    A.IMG_DIR = SF / "string_outputs_disease_specific/official_figures/query_only"
    C.SRC = D.SRC = pooled           # and every path the modules derived from SRC at import
    C.PNG = pooled / "POOLED12_query_only_official_STRING_v12.png"
    C.SVG = D.SVG = pooled / "POOLED12_query_only_official_STRING_v12.svg"
    D.PNG = pooled / "POOLED12_query_only_official_STRING_v12_readable.png"
    D.NETWORK = pooled / "network.tsv"
    D.PPI_TSV = pooled / "ppi_enrichment.tsv"
    # fig10_5_ppi_combined hard-codes the v4 edge counts under panels B-D.
    import pandas as pd
    st = pd.read_csv(STATS).set_index("panel")
    lab = {p: f"{int(st.at[p, 'official_number_of_edges'])} observed · "
              f"{st.at[p, 'official_expected_number_of_edges']:.0f} expected edges" for p in ("LOCKED25", "POOLED12")}
    F.PANEL_STATS = {"B": lab["LOCKED25"], "C": lab["POOLED12"], "D": lab["POOLED12"]}
    print("panel stats:", F.PANEL_STATS)

    for m in (A, B, C, D):
        m.main(lean=True)
    F.main()

    src = BUILD / "fig10_5_ppi_combined.png"
    ref = Image.open(REF).size
    im = Image.open(src)
    im.resize((ref[0], round(im.height * ref[0] / im.width)), Image.LANCZOS).save(LIVE, optimize=True)
    print(LIVE, "(v4.9 size", ref, ")")


if __name__ == "__main__":
    main()
