from typing import Any

import numpy as np
import pandas as pd


def make_hashable(x: Any):
    if x is pd.NA or x is None:
        return "__NaN__"

    if isinstance(x, (str, bytes, int, float, bool, np.generic)):
        try:
            if pd.isna(x):
                return "__NaN__"
        except Exception:
            pass

    if isinstance(x, np.generic):
        try:
            return x.item()
        except Exception:
            return str(x)

    if isinstance(x, np.ndarray):
        try:
            return tuple(make_hashable(v) for v in x.tolist())
        except Exception:
            return str(x)

    if isinstance(x, list):
        try:
            return tuple(make_hashable(v) for v in x)
        except Exception:
            return str(x)

    if isinstance(x, tuple):
        try:
            return tuple(make_hashable(v) for v in x)
        except Exception:
            return str(x)

    if isinstance(x, dict):
        try:
            return tuple(sorted((str(k), make_hashable(v)) for k, v in x.items()))
        except Exception:
            return str(x)

    if isinstance(x, set):
        try:
            return tuple(sorted(make_hashable(v) for v in x))
        except Exception:
            return str(x)

    try:
        if pd.isna(x):
            return "__NaN__"
    except Exception:
        pass

    return x


def safe_series(series: pd.Series) -> pd.Series:
    return series.map(make_hashable).astype("object")


def find_unhashable_columns(
    df: pd.DataFrame,
    sample_rows: int = 5000,
) -> list[str]:
    sub = df.head(sample_rows)
    bad_cols = []

    for col in sub.columns:
        for x in sub[col]:
            if isinstance(x, (np.ndarray, list, dict, set)):
                bad_cols.append(col)
                break

    return bad_cols


def afd_error_unary(df: pd.DataFrame, lhs: str, rhs: str) -> float:
    if lhs == rhs:
        return 0.0

    n = len(df)
    if n <= 1:
        return 0.0

    lhs_s = safe_series(df[lhs])
    rhs_s = safe_series(df[rhs])

    work = pd.DataFrame({lhs: lhs_s, rhs: rhs_s})

    sum_max = 0
    for _, group in work.groupby(lhs, dropna=False, sort=False):
        rhs_counts = group[rhs].value_counts(dropna=False, sort=False)
        sum_max += int(rhs_counts.max())

    return float(1.0 - sum_max / n)


def compute_best_fd_per_rhs(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    records = []

    for lhs in cols:
        for rhs in cols:
            if lhs == rhs:
                continue
            try:
                err = afd_error_unary(df, lhs, rhs)
            except Exception:
                continue

            records.append(
                {
                    "lhs": lhs,
                    "rhs": rhs,
                    "score": 1.0 - err,
                }
            )

    if not records:
        return pd.DataFrame(columns=["rhs", "best_lhs", "best_score"])

    pair_df = pd.DataFrame(records)

    best = (
        pair_df.sort_values(["rhs", "score"], ascending=[True, False])
        .groupby("rhs", as_index=False)
        .first()
        .rename(columns={"lhs": "best_lhs", "score": "best_score"})
    )

    return best[["rhs", "best_lhs", "best_score"]]