"""v7 Supplementary Figure: functional annotation (v6.x Figure 9 / figurev6 slide-6 mosaic), rebuilt with v7 inputs.
(A) ORA of the 26-protein panel against the measured proteins and the full annotation universe: unchanged
    figurev6/scripts/fig6_pathway_enrichment_v2.py, ORA folder -> v7 B9 output, panel key multiclass_26.
(B) Functional categories of the 81 disease-specific proteins: unchanged fig6_parta_sankey_bubble.py with the v7
    panels and the v5 category map extended by 13 proteins new in v7 (V7_FUNC; assigned from UniProt function,
    flagged for author review).
The v6.x chord panel (term links of the 25-protein set, from the legacy full-universe outputs) is not rebuilt.
Outputs: figures/output/figS_v7_pathway.png
"""
import re

import pandas as pd
from PIL import Image

import v7_style as S

SC = S.ROOT / "figurev6/scripts"
BUILD = S.V7 / "figures/build/figS_pathway"
V7_FUNC = {"ABHD14B": "Enzyme/Protease", "CALB2": "Neural/Struct.", "CD200": "Immune/Hematop.", "CD28": "Immune/Hematop.",
           "CRH": "Metabolic/Endocr.", "CXCL9": "Cytokine", "ELOA": "Signaling", "ERBB2": "Cancer marker",
           "NMNAT1": "Enzyme/Protease", "NT5E": "Enzyme/Protease", "PARP1": "Enzyme/Protease", "TGM2": "Enzyme/Protease",
           "XPNPEP2": "Enzyme/Protease"}


def v5_func():
    src = (S.ROOT / "overleaf-tools/make_fig9_v50_pathway.py").read_text(encoding="utf-8")
    block = src[src.index("V5_FUNC = {"):src.index("}", src.index("V5_FUNC = {")) + 1]
    return dict(re.findall(r'"([A-Z0-9]+)": "([^"]+)"', block))


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    # A: ORA bubbles
    src = (SC / "fig6_pathway_enrichment_v2.py").read_text(encoding="utf-8")
    for o, n in {'ORA = ROOT / "revision_package_2026-09-04" / "05_data_audit" / "ora_assay_background_2026-09-05"': f'ORA = Path(r"{S.RES / "B9_biology/ora"}")',
                 'OUT = ROOT / "figurev6" / "output"': f'OUT = Path(r"{BUILD}")',
                 't25 = t[t.panel == "multiclass_25"]': 't25 = t[t.panel == "multiclass_26"]',
                 # v7 colour rule (purple = 26-protein list only): GO in blue instead of purple
                 '("GO_BP_2023",      "GO Biological Process", "#7b3fa0", "#f4eefa", "RdPu")':
                     f'("GO_BP_2023",      "GO Biological Process", "{S.GO_COLOR}", "#eef3fa", "Blues")'}.items():
        assert src.count(o) == 1, o
        src = src.replace(o, n)
    ns = {"__file__": str(SC / "fig6_pathway_enrichment_v2.py"), "__name__": "figS_ora"}
    exec(compile(src, "fig6_pathway_enrichment_v2 (v7)", "exec"), ns)
    ns["main"]()
    # B: Sankey of disease-specific proteins -> categories
    pan = pd.read_csv(S.V7 / "panels/panel_memberships_v7.csv")
    deploy = {c: pan[pan.panel == ("single_DLBCL" if c == "LYMPH" else f"single_{c}")].protein.tolist() for c in S.FC.CANCER_ORDER}
    src = (SC / "fig6_parta_sankey_bubble.py").read_text(encoding="utf-8")
    old_load = src[src.index("si_files = list(PKG.rglob"):src.index("# ── edge table")]
    src = src.replace(old_load, "deploy = V7_DEPLOY\naucs = {c: None for c in CANCERS}\n\n")
    src = src.replace("FUNC_ORDER = [", "FUNC.update(V7_FUNC)\nFUNC_ORDER = [", 1)
    src = src.replace('OUT  = ROOT / "figurev6" / "output"', "OUT  = V7_BUILD")
    src = src.replace('f"fig6_parta_sankey.{ext}"', 'f"figS_v7_parta_sankey.{ext}"')
    funcs = {**v5_func(), **V7_FUNC}
    ns = {"__file__": str(SC / "fig6_parta_sankey_bubble.py"), "__name__": "figS_sankey", "V7_DEPLOY": deploy,
          "V7_FUNC": funcs, "V7_BUILD": BUILD}
    exec(compile(src, "fig6_parta_sankey_bubble (v7)", "exec"), ns)
    missing = sorted({p for v in deploy.values() for p in v} - set(ns["FUNC"]))
    assert not missing, missing
    import build_slide_mosaics as bsm
    bsm.OUT = BUILD
    spec = dict(bsm.SLIDES["fig6_slide6_pathway_mosaic"])
    spec["grid"] = (2, 1)
    spec["panels"] = [("fig6_pathway_enrichment", 0, 0, 1, 1), ("figS_v7_parta_sankey", 1, 0, 1, 1)]
    bsm.build("figS_v7_pathway", spec)
    Image.open(BUILD / "figS_v7_pathway.png").save(S.OUT / "figS_v7_pathway.png", optimize=True)
    print("wrote", S.OUT / "figS_v7_pathway.png")


if __name__ == "__main__":
    main()
