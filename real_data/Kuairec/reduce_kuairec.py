"""Reduce the official KuaiRec release to the files used by the analysis (columns kept verbatim)"""
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from real_data.kuairec.paths import DATA_RAW, DATA_RED

COLS = ["user_id", "video_id", "play_duration", "video_duration", "timestamp", "watch_ratio"]
MD5 = "261550d472c48eff4990fb13c0e5bcf7"


def main(zpath):
    zpath = Path(zpath)
    h = hashlib.md5()
    with open(zpath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    print("md5", h.hexdigest(), "(expected", MD5 + ")")
    DATA_RED.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        d = "KuaiRec 2.0/data/"
        S = pd.read_csv(z.open(d + "small_matrix.csv"), usecols=COLS)
        np.savez_compressed(DATA_RED / "small_matrix.npz", **{c: S[c].values for c in COLS})
        users = np.unique(S.user_id.values)
        parts = [ch[ch.user_id.isin(users)] for ch in
                 pd.read_csv(z.open(d + "big_matrix.csv"), usecols=COLS, chunksize=2_000_000)]
        B = pd.concat(parts)
        np.savez_compressed(DATA_RED / "big_matrix_smallusers.npz", **{c: B[c].values for c in COLS})
        for f in ("item_categories.csv", "user_features.csv"):
            (DATA_RED / f).write_bytes(z.read(d + f))
    print("small matrix", S.shape, " big matrix rows of small-matrix users", B.shape)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_RAW / "KuaiRec.zip")
