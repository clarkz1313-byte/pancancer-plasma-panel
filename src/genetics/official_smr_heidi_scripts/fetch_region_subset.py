#!/usr/bin/env python3
"""Fetch one indexed BGZF region and write a 503-person European VCF.

The source byte range, response headers, and hashes are recorded. This script
only prepares reference genotypes. It does not implement any SMR or HEIDI
statistic.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import struct
import urllib.request
import zlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_tbi_offset(path: Path, chrom: str, position: int) -> tuple[int, int]:
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    offset = 4

    def i32() -> int:
        nonlocal offset
        value = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        return value

    def u64() -> int:
        nonlocal offset
        value = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        return value

    if data[:4] != b"TBI\x01":
        raise ValueError(f"Invalid TBI magic in {path}")
    n_ref = i32()
    for _ in range(6):
        i32()
    names_len = i32()
    names = data[offset : offset + names_len].decode().split("\0")[:-1]
    offset += names_len
    if chrom not in names:
        raise ValueError(f"Chromosome {chrom} absent from {path}")
    target_index = names.index(chrom)
    linear = None
    for ref_index in range(n_ref):
        n_bin = i32()
        for _ in range(n_bin):
            i32()
            n_chunk = i32()
            offset += 16 * n_chunk
        n_interval = i32()
        values = struct.unpack_from(f"<{n_interval}Q", data, offset)
        offset += 8 * n_interval
        if ref_index == target_index:
            linear = values
            break
    if linear is None:
        raise ValueError(f"No linear index for chromosome {chrom}")
    index = min((position - 1) // 16384, len(linear) - 1)
    virtual_offset = linear[index]
    while virtual_offset == 0 and index > 0:
        index -= 1
        virtual_offset = linear[index]
    return virtual_offset >> 16, virtual_offset & 0xFFFF


def fetch_range(url: str, start: int, size: int) -> tuple[bytes, dict[str, str], int]:
    request = urllib.request.Request(url)
    request.add_header("Range", f"bytes={start}-{start + size - 1}")
    with urllib.request.urlopen(request, timeout=180) as response:
        status = response.status
        headers = {key.lower(): value for key, value in response.headers.items()}
        payload = response.read()
    if status != 206:
        raise RuntimeError(f"Expected HTTP 206 for byte range, received {status}")
    content_range = headers.get("content-range", "")
    if not content_range.startswith(f"bytes {start}-"):
        raise RuntimeError(f"Unexpected Content-Range: {content_range!r}")
    return payload, headers, status


def decompress_bgzf(payload: bytes) -> tuple[bytes, int, int]:
    offset = 0
    decoded = bytearray()
    members = 0
    payload_size = len(payload)
    while offset + 18 <= payload_size:
        if payload[offset : offset + 2] != b"\x1f\x8b":
            break
        block_size = struct.unpack_from("<H", payload, offset + 16)[0] + 1
        if offset + block_size > payload_size:
            break
        try:
            decoded.extend(gzip.decompress(payload[offset : offset + block_size]))
        except (gzip.BadGzipFile, EOFError, zlib.error):
            break
        offset += block_size
        members += 1
    return bytes(decoded), members, offset


def read_eur_samples(panel_path: Path) -> set[str]:
    with panel_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["sample"] for row in reader if row["super_pop"] == "EUR"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--tbi", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--range-mb", type=int, default=80)
    args = parser.parse_args()

    all_samples = [line.strip() for line in args.samples.read_text().splitlines() if line.strip()]
    eur = read_eur_samples(args.panel)
    selected = [(index, sample) for index, sample in enumerate(all_samples) if sample in eur]
    if len(selected) != 503:
        raise RuntimeError(f"Expected 503 European samples, found {len(selected)}")

    index_chrom = args.chrom
    try:
        compressed_offset, uncompressed_offset = read_tbi_offset(
            args.tbi, index_chrom, args.start
        )
    except ValueError:
        index_chrom = f"chr{args.chrom}"
        compressed_offset, uncompressed_offset = read_tbi_offset(
            args.tbi, index_chrom, args.start
        )
    payload, headers, status = fetch_range(
        args.url, compressed_offset, args.range_mb * 1024 * 1024
    )
    decoded, member_count, consumed = decompress_bgzf(payload)
    text = decoded.decode("utf-8", errors="strict")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    variants_seen = 0
    variants_written = 0
    reached_end = False
    lead_positions = {}
    with args.out.open("w", encoding="utf-8", newline="\n") as out:
        out.write("##fileformat=VCFv4.2\n")
        out.write(f"##contig=<ID={args.chrom}>\n")
        out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        out.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(sample for _, sample in selected)
            + "\n"
        )
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9 + len(all_samples):
                continue
            try:
                position = int(fields[1])
            except ValueError:
                continue
            if position < args.start:
                continue
            if position > args.end:
                reached_end = True
                break
            variants_seen += 1
            ref, alt = fields[3].upper(), fields[4].upper()
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            canonical_id = f"{args.chrom}:{position}:{ref}:{alt}"
            core = [args.chrom, str(position), canonical_id, ref, alt, ".", "PASS", ".", "GT"]
            genotypes = [fields[9 + index].split(":", 1)[0] for index, _ in selected]
            out.write("\t".join(core + genotypes) + "\n")
            variants_written += 1
            lead_positions[str(position)] = canonical_id

    if not reached_end:
        raise RuntimeError(
            f"Fetched range did not reach {args.chrom}:{args.end}; increase --range-mb"
        )

    provenance = {
        "source_url": args.url,
        "http_status": status,
        "content_range": headers.get("content-range"),
        "requested_compressed_offset": compressed_offset,
        "tbi_uncompressed_offset": uncompressed_offset,
        "downloaded_bytes": len(payload),
        "downloaded_sha256": sha256_bytes(payload),
        "decoded_bgzf_members": member_count,
        "decoded_compressed_bytes": consumed,
        "region": f"{args.chrom}:{args.start}-{args.end}",
        "tabix_chromosome_name": index_chrom,
        "source_sample_count": len(all_samples),
        "retained_eur_sample_count": len(selected),
        "biallelic_snv_count": variants_written,
        "all_variants_in_region": variants_seen,
        "output_vcf": str(args.out),
        "output_vcf_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
