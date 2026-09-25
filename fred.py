"""FRED series via the public fredgraph CSV endpoint (no API key needed)."""
from __future__ import annotations

import io

import pandas as pd

from utils import get


def series(ids: list[str]) -> pd.DataFrame:
    out = []
    for sid in ids:
        r = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
        s = pd.read_csv(io.StringIO(r.text), na_values=["."])
        date_col = [c for c in s.columns if c.lower() in ("date", "observation_date")][0]
        s[date_col] = pd.to_datetime(s[date_col])
        out.append(s.set_index(date_col)[sid].astype(float))
    return pd.concat(out, axis=1)
