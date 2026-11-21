# v7 FORK of revise_plan/scripts/56 (original unchanged): v7 disease-specific panels; outputs under v7_plan/results/B9_biology/string_figures.
#!/usr/bin/env python3
"""
v5.0 Figure 10 inputs: official STRING v12 network figures for the v5 panels.

1. Per-panel networks: runs revise_plan/ppi_string/batch_string_export.py
   unchanged except EXPECTED_COUNTS (its hard-coded v4 panel sizes), which is
   replaced by the v5 sizes, on biology/disease_specific_panels_v5.csv.
2. Pooled network: runs revise_plan/ppi_string/fetch_pooled12panels_string.py
   with PANELS_CSV -> the v5 panel file and OUT -> the v5 folder.
Both write only under revise_plan/v5_seed52_common_split/biology/string_figures/.
Enrichment statistics are not taken from here (they come from script 47).
"""
import os as _os
from pathlib import Path as _Path
PROJECT_ROOT = _Path(_os.environ.get("PROTEOMICS_ROOT", _Path(__file__).resolve().parents[2]))
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd

PPI = (PROJECT_ROOT / "revise_plan/ppi_string")
V5B = (PROJECT_ROOT / "v7_plan/results/B9_biology")
PANELS = V5B / "inputs/disease_specific_panels_v7.csv"
OUT = V5B / "string_figures"


def run_patched(script: Path, replacements: dict, argv: list, extra: dict | None = None) -> None:
    src = script.read_text(encoding="utf-8")
    for old, new in replacements.items():
        assert src.count(old) == 1, (script.name, old)
        src = src.replace(old, new)
    ns = {"__file__": str(script), "__name__": "__main__"}
    ns.update(extra or {})
    old_argv = sys.argv
    sys.argv = [str(script)] + argv
    try:
        exec(compile(src, f"{script.name} (v5 patched)", "exec"), ns)
    except SystemExit as e:  # batch_string_export ends with sys.exit(main())
        if e.code not in (0, None):
            raise
    finally:
        sys.argv = old_argv


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = pd.read_csv(PANELS)
    counts = OrderedDict((p, len(g.split(";"))) for p, g in zip(panels.Panel, panels.Genes))
    print("v5 panel sizes:", dict(counts))

    src = (PPI / "batch_string_export.py").read_text(encoding="utf-8")
    start = src.index("EXPECTED_COUNTS = OrderedDict(")
    end = src.index("\n)\n", start) + 3
    run_patched(PPI / "batch_string_export.py", {src[start:end]: "EXPECTED_COUNTS = V5_COUNTS\n"},
                ["--panels", str(PANELS), "--output", str(OUT / "string_outputs_disease_specific"), "--no-zip"],
                {"V5_COUNTS": counts})

    run_patched(PPI / "fetch_pooled12panels_string.py",
                {'PANELS_CSV = HERE / "disease_specific_panels.csv"': "PANELS_CSV = V5_PANELS",
                 'OUT = HERE / "STRING_analysis_pooled12panels"': "OUT = V5_POOLED"},
                [], {"V5_PANELS": PANELS, "V5_POOLED": OUT / "STRING_analysis_pooled12panels"})


if __name__ == "__main__":
    main()
