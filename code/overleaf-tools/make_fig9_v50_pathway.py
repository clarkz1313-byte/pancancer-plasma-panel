"""Manuscript v5.0 Figure 9 (functional annotations; figurev6 slide-6 mosaic).

Panels A and B describe the 25-protein set, which v5 does not change, so the
existing figurev6/output images are reused (copied read-only). Panel C (Sankey of
disease-specific proteins -> function category) is re-rendered by executing the
unchanged figurev6/scripts/fig6_parta_sankey_bubble.py with three patches:
  1. panel membership read from the v5 canonical file (was: locked25 results
     package, seed-42 panels);
  2. function categories added for the v5 proteins absent from its FUNC map
     (V5_FUNC below; curated from UniProt function, flagged for author review);
  3. output written to the v5 build folder under a new name.
Output: overleaf-source-git/figures_300/fig9_v50_pathway_mosaic.png (new file).
"""
from pathlib import Path
import shutil
import sys

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "revise_plan/v5_seed52_common_split"
BUILD = V5 / "figure_inputs/fig9_build"
SCRIPTS = ROOT / "figurev6/scripts"
FIG6OUT = ROOT / "figurev6/output"
REF = ROOT / "overleaf-source-git/figures_300/fig6_slide6_pathway_mosaic.png"
LIVE = ROOT / "overleaf-source-git/figures_300/fig9_v50_pathway_mosaic.png"
sys.path.insert(0, str(SCRIPTS))

# Categories for disease-specific proteins new in v5 (same category scheme).
# Checked against UniProt (reviewed, human) function text and keywords on
# 2026-09-30; AGR2, CASP8 and DNER were re-assigned after that check.
V5_FUNC = {
    "AGR2": "Cancer marker",        # UniProt KW proto-oncogene; tumour-associated secreted protein
    "ANGPTL2": "Growth factor",     # angiopoietin-like 2; induces endothelial sprouting
    "CASP8": "Enzyme/Protease",     # UniProt KW protease, thiol protease (apoptosis initiator)
    "CD74": "Immune/Hematop.",      # MHC class II invariant chain
    "DNER": "Signaling",            # UniProt KW receptor, Notch signalling pathway
    "F2R": "Signaling",             # thrombin receptor PAR-1
    "FCN2": "Immune/Hematop.",      # ficolin-2, lectin complement pathway
    "FGF23": "Growth factor",       # fibroblast growth factor 23
    "GMPR": "Enzyme/Protease",      # GMP reductase
    "GUCA2A": "Metabolic/Endocr.",  # guanylin, intestinal peptide hormone
    "GZMB": "Immune/Hematop.",      # granzyme B, cytotoxic lymphocytes
    "IL10": "Cytokine",
    "IL17A": "Cytokine",
    "LTBP2": "ECM/Adhesion",        # latent TGF-beta binding protein 2
    "NCS1": "Neural/Struct.",       # neuronal calcium sensor 1
    "PCSK9": "Metabolic/Endocr.",   # LDL-receptor regulation
    "PRL": "Metabolic/Endocr.",     # prolactin
    "SFTPD": "Immune/Hematop.",     # surfactant protein D, collectin
    "SMAD5": "Signaling",           # BMP signalling
    "SPINT1": "Enzyme/Protease",    # HAI-1 serine protease inhibitor
    "TNFRSF9": "Immune/Hematop.",   # CD137
}


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    hub = pd.read_csv(V5 / "panels/panel_memberships_v5.csv")
    deploy = {}
    for code in ["AML", "BRC", "CLL", "CRC", "CVX", "ENDC", "GLIOM", "LUNGC", "LYMPH", "MYEL", "OVC", "PRC"]:
        key = "single_DLBCL" if code == "LYMPH" else f"single_{code}"
        deploy[code] = hub[hub.panel == key].protein.tolist()
        assert deploy[code], code

    src = (SCRIPTS / "fig6_parta_sankey_bubble.py").read_text(encoding="utf-8")
    old_load = src[src.index("si_files = list(PKG.rglob"):src.index("# ── edge table")]
    src = src.replace(old_load, "deploy = V5_DEPLOY\naucs = {c: None for c in CANCERS}\n\n")
    src = src.replace('FUNC_ORDER = [', 'FUNC.update(V5_FUNC)\nFUNC_ORDER = [', 1)
    src = src.replace('OUT  = ROOT / "figurev6" / "output"', 'OUT  = V5_BUILD')
    src = src.replace('f"fig6_parta_sankey.{ext}"', 'f"fig6_v50_parta_sankey.{ext}"')
    ns = {"__file__": str(SCRIPTS / "fig6_parta_sankey_bubble.py"), "__name__": "fig6_v50",
          "V5_DEPLOY": deploy, "V5_FUNC": V5_FUNC, "V5_BUILD": BUILD}
    exec(compile(src, "fig6_parta_sankey_bubble.py (v50 patched)", "exec"), ns)

    all_prot = {p for v in deploy.values() for p in v}
    missing = sorted(all_prot - set(ns["FUNC"]))
    assert not missing, f"no category for {missing}"
    print("category counts:", pd.Series({p: ns["FUNC"][p] for p in all_prot}).value_counts().to_dict())

    import build_slide_mosaics as bsm
    for a in ("fig6_pathway_enrichment", "fig6_chord_network"):
        shutil.copyfile(FIG6OUT / f"{a}.png", BUILD / f"{a}.png")
    bsm.OUT = BUILD
    spec = dict(bsm.SLIDES["fig6_slide6_pathway_mosaic"]) if hasattr(bsm, "SLIDES") else None
    if spec is None:
        raise SystemExit("build_slide_mosaics has no SLIDES dict; adapt the spec lookup")
    spec["panels"] = [("fig6_pathway_enrichment", 0, 0, 1, 2), ("fig6_chord_network", 1, 0, 1, 1),
                      ("fig6_v50_parta_sankey", 1, 1, 1, 1)]
    stem = "fig9_v50_pathway_mosaic"
    bsm.build(stem, spec)
    ref = Image.open(REF).size
    im = Image.open(BUILD / f"{stem}.png")
    im.resize((ref[0], round(im.height * ref[0] / im.width)), Image.LANCZOS).save(LIVE, optimize=True)
    print(LIVE, "(v4.9 size", ref, ")")


if __name__ == "__main__":
    main()
