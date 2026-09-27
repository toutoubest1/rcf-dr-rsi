"""Minimal ASlib reader (ASlib format specification, coseal/aslib_data).

Reads description.txt (YAML), algorithm_runs.arff, feature_values.arff and feature_runstatus.arff.
ARFF parsing: attributes in declared order, data section as CSV ('?' = missing). No third-party ARFF library is
used because instance ids are STRING attributes (not supported by scipy.io.arff).
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from real_data.aslib.paths import DATA


def read_arff(path: Path) -> pd.DataFrame:
    attrs, types, data_lines, in_data = [], [], [], False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if in_data:
                data_lines.append(s)
                continue
            low = s.lower()
            if low.startswith("@attribute"):
                rest = s.split(None, 1)[1]
                if rest.startswith("'") or rest.startswith('"'):
                    q = rest[0]
                    j = rest.index(q, 1)
                    name, typ = rest[1:j], rest[j + 1:].strip()
                else:
                    name, typ = rest.split(None, 1)
                attrs.append(name)
                types.append(typ.lower())
            elif low.startswith("@data"):
                in_data = True
    rows = list(csv.reader(io.StringIO("\n".join(data_lines)), quotechar="'", skipinitialspace=True))
    df = pd.DataFrame(rows, columns=attrs)
    for a, t in zip(attrs, types):
        if t.startswith("numeric") or t.startswith("real") or t.startswith("integer"):
            df[a] = pd.to_numeric(df[a].replace("?", np.nan), errors="coerce")
        else:
            df[a] = df[a].str.strip().str.strip("'\"")
    return df


def load_scenario(name: str, root: Path = DATA):
    d = root / name
    desc = yaml.safe_load(open(d / "description.txt"))
    runs = read_arff(d / "algorithm_runs.arff")
    feats = read_arff(d / "feature_values.arff")
    return desc, runs, feats
