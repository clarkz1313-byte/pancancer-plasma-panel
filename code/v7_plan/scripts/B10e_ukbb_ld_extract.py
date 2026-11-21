"""v7 B10e helper: extract a UK Biobank LD submatrix (Weissbrod et al. 2020, Broad public UKBB_LD, GRCh37,
3-Mb windows; scipy sparse upper triangle, as read by polyfun's load_ld_npz) for a list of SNPs.

Usage: python B10e_ukbb_ld_extract.py <window prefix> <snps.csv with pos37,a1,a0> <out prefix>
Matches on GRCh37 position and both alleles (direct or swapped; strand complement accepted). Writes
<out>_matched.csv (input row, pos37, orient = +1 when the LD allele1 is the pQTL effect allele a1, -1 when swapped)
and <out>_R.npy (correlation matrix in the order of the matched rows).
"""
import sys

import numpy as np
import pandas as pd
from scipy import sparse

COMP = str.maketrans("ACGT", "TGCA")


def main(prefix, snps_csv, out):
    info = pd.read_csv(prefix + ".gz", sep=r"\s+")
    info = info.rename(columns={"allele1": "A1", "allele2": "A2", "position": "BP"})
    info["ld_idx"] = np.arange(len(info))
    s = pd.read_csv(snps_csv)
    s["row"] = np.arange(len(s))
    m = s.merge(info[["BP", "A1", "A2", "ld_idx"]], left_on="pos37", right_on="BP")
    a1, a0 = m.a1.str.upper(), m.a0.str.upper()
    A1, A2 = m.A1.str.upper(), m.A2.str.upper()
    c1, c0 = a1.str.translate(COMP), a0.str.translate(COMP)
    m["orient"] = np.select([(A1 == a1) & (A2 == a0), (A1 == a0) & (A2 == a1), (A1 == c1) & (A2 == c0), (A1 == c0) & (A2 == c1)],
                            [1, -1, 1, -1], 0)
    m = m[m.orient != 0].drop_duplicates("row").sort_values("ld_idx")
    M = sparse.load_npz(prefix + ".npz").tocsr()
    idx = m.ld_idx.to_numpy()
    sub = M[idx][:, idx].toarray()
    R = sub + sub.T
    assert np.allclose(np.diag(R), 1.0), "unexpected LD diagonal"
    m[["row", "pos37", "orient", "ld_idx"]].to_csv(out + "_matched.csv", index=False)
    np.save(out + "_R.npy", R)
    np.savetxt(out + "_R.csv", R, delimiter=",", fmt="%.6f")
    print(f"window SNPs {len(info)}, requested {len(s)}, matched {len(m)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
