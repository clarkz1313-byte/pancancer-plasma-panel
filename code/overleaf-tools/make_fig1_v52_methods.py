"""Correct labels in the editable Figure 1 SVG without changing analysis outputs.

Retains the existing layout and organ illustrations. Replaces the embedded
tile labels with vector text from Table S5, clarifies the two selection
branches, and replaces the neural-network illustration with the actual model.
Render the resulting SVG in a browser so draw.io's HTML labels are preserved.
"""
from pathlib import Path
import csv
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SVG = "http://www.w3.org/2000/svg"
HTML = "http://www.w3.org/1999/xhtml"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xhtml", HTML)
ET.register_namespace("xlink", XLINK)
NS = {"s": SVG}


def add(parent, tag, **attrs):
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", {k: str(v) for k, v in attrs.items()})


def label(parent, x, y, lines, size=12, weight="bold", spacing=14):
    text = add(parent, "text", x=x, y=y, fill="#000000",
               **{"font-family": "Arial, Helvetica, sans-serif", "font-size": size,
                  "font-weight": weight, "text-anchor": "middle"})
    for i, line in enumerate(lines):
        add(text, "tspan", x=x, y=y + i * spacing).text = line
    return text


def main():
    src = ROOT / "overleaf-source-git/figures_600/fig1_v52.svg"
    tree = ET.parse(src)
    root = tree.getroot()
    parents = {child: parent for parent in root.iter() for child in parent}
    replacements = {
        "Feature ranking": (305, 373, ["Differential", "abundance", "screening"], 12),
        "Disease-specific": (469, 325, ["LASSO ranking", "Test-set size choice", "Final logistic model"], 10.5),
        "Multi-Classification\nCandidate bank": (469, 435, ["Multiclass", "Candidate bank", "801 to 180 proteins"], 12),
        "Multi-Classification\nPanels assembly": (751, 439, ["3,000 panels", "CV shortlist: 540", "Test-set choice"], 11),
        "Multi classification panel": (458, 130, ["25-protein multiclass panel"], 11),
        "External transferability protein": (712, 28, ["External marker-set refits", "Protein and transcript data"], 11),
        "AML, BRC, CLL, CRC,": (260, 89, ["AML, BRC, CLL, CRC, CVX,", "ENDC, GLIOM, LUNGC,", "DLBCL, MYEL, OVC, PRC"], 9),
    }
    done = set()
    for text in list(root.findall(".//s:text", NS)):
        lines = ["".join(e.itertext()) for e in list(text)] or ["".join(text.itertext())]
        content = "\n".join(lines)
        for prefix, (x, y, values, size) in replacements.items():
            if content.startswith(prefix):
                parent = parents[text]
                # The sibling foreignObject contains the same label in HTML.
                for child in list(parent):
                    if child.tag in (f"{{{SVG}}}foreignObject", f"{{{SVG}}}text"):
                        parent.remove(child)
                label(parent, x, y, values, size=size, weight="normal" if prefix.startswith("AML,") else "bold")
                done.add(prefix)
                break
    assert done == set(replacements), done

    # The original tile artwork is an embedded bitmap. Only its text regions
    # are covered by vector rectangles and labels; all organ icons are retained.
    with (ROOT / "overleaf-source-git/supplement/v5.1/Table_S5_disease_specific_panel_membership.csv").open(encoding="utf-8-sig", newline="") as fh:
        sizes = {row["cancer_code"]: int(row["panel_size"]) for row in csv.DictReader(fh)}
    overlay = add(root, "g", id="current-panel-labels")
    rows = [["AML", "BRC", "CLL", "CRC"], ["GLIOM", "LUNGC", "DLBCL", "MYEL"], ["CVX", "ENDC", "OVC", "PRC"]]
    centres = [3780, 4274, 4768, 5260]
    for codes, code_y, count_y in zip(rows, [1558, 1848, 2138], [1621, 1911, 2201]):
        for code, x in zip(codes, centres):
            add(overlay, "rect", x=x-130, y=code_y-57, width=260, height=155, fill="#ffffff")
            label(overlay, x, code_y, [code], size=54, spacing=64)
            label(overlay, x, count_y, [f"{sizes[code]} proteins"], size=40, weight="normal")

    # The final estimator is logistic regression, not a neural network.
    neural = [e for e in root.findall(".//s:image", NS)
              if e.get("x") == "5128.37" and e.get("width") == "600"]
    assert len(neural) == 1
    parents[neural[0]].remove(neural[0])
    add(overlay, "rect", x=5140, y=2530, width=570, height=292, rx=42,
        fill="#eef4fd", stroke="#7fa7e2", **{"stroke-width": 6})
    label(overlay, 5425, 2630, ["L2 multinomial", "logistic regression"], size=51, spacing=70)

    out = src.with_name("fig1_v52_methods.svg")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(out)
    print("Panel sizes:", sizes)

    import subprocess
    import tempfile
    chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    png = out.with_suffix(".png")
    subprocess.run([
        str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--no-sandbox", "--no-first-run", "--disable-extensions",
        "--force-device-scale-factor=1", "--default-background-color=ffffffff",
        "--window-size=5766,3129",
        "--user-data-dir=" + tempfile.mkdtemp(prefix="proteomics_fig1_render_"),
        "--screenshot=" + str(png), out.as_uri(),
    ], check=True, capture_output=True, timeout=60)
    print(png)


if __name__ == "__main__":
    main()
