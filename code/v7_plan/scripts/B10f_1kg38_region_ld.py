"""v7 B10f helper: GRCh38 LD from 1000 Genomes high-coverage phased genotypes (NYGC 3,202-sample panel), European
founders only (Superpopulation EUR, no parents in the pedigree file), for one region, fetched by HTTP range
requests using the tabix linear index (no whole-chromosome download).

Usage: python B10f_1kg38_region_ld.py <chrom> <start> <end> <snps.csv with pos38,a1,a0> <out prefix>
Matches SNPs on GRCh38 position and both alleles (biallelic records only). Writes <out>_matched.csv (row, pos38,
orient = +1 if the VCF ALT allele is the pQTL effect allele a1, -1 if it is a0) and <out>_R.csv (correlation of
ALT dosages, haplotype sums, in the order of the matched rows).
"""
import gzip
import io
import struct
import sys
import urllib.request
import zlib

import numpy as np
import pandas as pd

BASE = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/"
        "20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr{c}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz")
PED = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt"


def get(url, a=None, b=None):
    h = {"Range": f"bytes={a}-{b}"} if a is not None else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300) as r:
        return r.read()


def linear_offsets(tbi):
    d = gzip.decompress(tbi)
    assert d[:4] == b"TBI\x01"
    n_ref, = struct.unpack("<i", d[4:8])
    l_nm, = struct.unpack("<i", d[32:36])
    p = 36 + l_nm
    assert n_ref == 1
    n_bin, = struct.unpack("<i", d[p:p + 4]); p += 4
    for _ in range(n_bin):
        _, n_chunk = struct.unpack("<Ii", d[p:p + 8]); p += 8 + 16 * n_chunk
    n_intv, = struct.unpack("<i", d[p:p + 4]); p += 4
    return struct.unpack(f"<{n_intv}Q", d[p:p + 8 * n_intv])


def bgzf_decompress(raw):
    out, i = [], 0
    while i < len(raw) - 18:
        bsize = struct.unpack("<H", raw[i + 16:i + 18])[0] + 1
        blk = raw[i:i + bsize]
        if len(blk) < bsize:
            break
        out.append(zlib.decompress(blk[18:-8], -15))
        i += bsize
    return b"".join(out)


def main(chrom, start, end, snps_csv, out):
    start, end = int(start), int(end)
    url = BASE.format(c=chrom)
    ioff = linear_offsets(get(url + ".tbi"))
    a = ioff[start >> 14] >> 16
    e_idx = min(len(ioff) - 1, (end >> 14) + 2)
    b = (ioff[e_idx] >> 16) + 70000
    text = bgzf_decompress(get(url, a, b)).decode()
    lines = text.split("\n")[1:]  # first line may be partial
    ped = pd.read_csv(io.BytesIO(get(PED)), sep=r"\s+")
    founders = set(ped[(ped.Superpopulation == "EUR") & (ped.FatherID.astype(str) == "0") & (ped.MotherID.astype(str) == "0")].SampleID)
    hdr = pd.read_csv(io.BytesIO(gzip.decompress(get(url, 0, 200000))[:0] or b""), sep="\t") if False else None
    # sample order from the VCF header (first BGZF blocks)
    head = bgzf_decompress(get(url, 0, 400000)).decode(errors="ignore")
    cols = [l for l in head.split("\n") if l.startswith("#CHROM")][0].split("\t")
    keep = [i for i, s in enumerate(cols) if s in founders]
    s = pd.read_csv(snps_csv)
    want = {int(p): i for i, p in enumerate(s.pos38)}
    rows, G = [], []
    for l in lines:
        if not l or l.startswith("#"):
            continue
        f = l.split("\t", 9)
        pos = int(f[1])
        if pos > end:
            break
        if pos < start or pos not in want or "," in f[4]:
            continue
        ref, alt = f[3].upper(), f[4].upper()
        i = want[pos]
        a1, a0 = s.a1[i].upper(), s.a0[i].upper()
        orient = 1 if (alt == a1 and ref == a0) else -1 if (alt == a0 and ref == a1) else 0
        if orient == 0:
            continue
        gt = l.split("\t")
        dose = np.array([sum(int(x) for x in gt[k].split(":")[0].replace("|", "/").split("/") if x in "01") for k in keep], float)
        if dose.std() == 0:
            continue
        rows.append((i, pos, orient)); G.append(dose)
    m = pd.DataFrame(rows, columns=["row", "pos38", "orient"]).drop_duplicates("row")
    G = np.array(G)[m.index.to_numpy()]
    R = np.corrcoef(G)
    m.to_csv(out + "_matched.csv", index=False)
    np.savetxt(out + "_R.csv", R, delimiter=",", fmt="%.6f")
    print(f"EUR founders {len(keep)}; requested {len(s)}, matched {len(m)}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
