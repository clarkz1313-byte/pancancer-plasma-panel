#!/usr/bin/env python3
"""Range-extract only the required GCTB 10K European LD blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


URL = "https://gctbhub.cloud.edu.au/data/SBayesRC/resources/GWFM/LD/Imputed13M/ukbEUR_13M_FullLDM.zip"
MEMBERS = (
    "ldm13M/ldm.info",
    "ldm13M/snp.info",
    "ldm13M/block1048.ldm.bin",
    "ldm13M/block1350.ldm.bin",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-dir", type=Path, required=True)
    args = parser.parse_args()
    tools = args.workflow_dir / "tools" / "python"
    sys.path.insert(0, str(tools))
    from remotezip import RemoteZip

    output = args.workflow_dir / "larger_ld" / "gctb_10k_eur"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with RemoteZip(URL) as archive:
        for member in MEMBERS:
            destination = output / Path(member).name
            with archive.open(member) as source, destination.open("wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
            info = archive.getinfo(member)
            records.append({
                "source_zip_url": URL,
                "member": member,
                "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "destination": str(destination),
                "sha256": sha256(destination),
            })
            print(f"Extracted {member} to {destination}", flush=True)
    manifest = output / "extraction_manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
