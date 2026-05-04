# -*- coding: utf-8 -*-

from rdblearn.datasets import RDBDataset
import pandas as pd
import numpy as np
from itertools import combinations
import os
import sys
import traceback


############################################
# BASIC STATS
############################################

def distinct_ratio(df: pd.DataFrame) -> pd.Series:
    return df.nunique(dropna=False) / len(df)


def is_hashable_value(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False


def find_unhashable_columns(df: pd.DataFrame):
    bad_cols = []
    for col in df.columns:
        sample = df[col].dropna().head(20)

        if len(sample) == 0:
            continue

        ok = True
        for x in sample:
            try:
                hash(x)
            except TypeError:
                ok = False
                break

        if not ok:
            bad_cols.append(col)

    return bad_cols


def row_dup_ratio(df: pd.DataFrame):
    if len(df) == 0:
        return np.nan

    usable_cols = []
    for col in df.columns:
        sample = df[col].dropna().head(20)

        if len(sample) == 0:
            usable_cols.append(col)
            continue

        if sample.map(is_hashable_value).all():
            usable_cols.append(col)

    if len(usable_cols) == 0:
        return np.nan

    return 1 - (len(df[usable_cols].drop_duplicates()) / len(df))


############################################
# FD SCORE
############################################

def fd_score(df, lhs, rhs):
    try:
        grouped = df.groupby(lhs)[rhs].nunique()
    except Exception:
        return None, None, None

    total = len(grouped)
    if total == 0:
        return None, None, None

    violations = int((grouped > 1).sum())
    score = 1 - (violations / total)
    return score, total, violations


############################################
# ANALYSIS
############################################

def analyze_table_basic(df, table_name):
    print("\n" + "=" * 100)
    print(f"TABLE: {table_name}")
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    bad_cols = find_unhashable_columns(df)
    print("\n[Unhashable columns]")
    print(bad_cols)

    try:
        print("row_dup_ratio:", row_dup_ratio(df))
    except Exception as e:
        print("row_dup_ratio FAILED:", e)

    try:
        dr = distinct_ratio(df)

        print("\n[Low distinct ratio columns]")
        print(dr.sort_values().head(10))

        print("\n[High distinct ratio columns]")
        print(dr.sort_values(ascending=False).head(10))

        print("\n[Key-like columns]")
        print(dr[dr > 0.99])
    except Exception as e:
        print("distinct_ratio FAILED:", e)


def analyze_fd(df, table_name):
    print("\n[Single-column FD candidates in", table_name, "]")

    cols = list(df.columns)

    strong = []
    moderate = []

    for lhs in cols:
        for rhs in cols:
            if lhs == rhs:
                continue

            try:
                score, n_groups, violations = fd_score(df, lhs, rhs)
            except Exception:
                continue

            if score is None:
                continue

            result = {
                "lhs": lhs,
                "rhs": rhs,
                "score": score,
                "n_groups": n_groups,
                "violating_groups": violations
            }

            if score >= 0.99:
                strong.append(result)
            elif score >= 0.95:
                moderate.append(result)

    strong = sorted(strong, key=lambda x: (-x["score"], str(x["lhs"]), str(x["rhs"])))
    moderate = sorted(moderate, key=lambda x: (-x["score"], str(x["lhs"]), str(x["rhs"])))

    print("\nStrong FD candidates (score >= 0.99):")
    for r in strong[:30]:
        print(r)

    print("\nModerate FD candidates (0.95 <= score < 0.99):")
    for r in moderate[:30]:
        print(r)

    print("\n[Two-column FD candidates in", table_name, "]")

    strong_pair = []
    moderate_pair = []

    # 컬럼 수가 너무 많으면 pair FD는 오래 걸릴 수 있어서 제한
    pair_cols = cols[:10]

    for lhs in combinations(pair_cols, 2):
        lhs = list(lhs)
        for rhs in pair_cols:
            if rhs in lhs:
                continue

            try:
                score, n_groups, violations = fd_score(df, lhs, rhs)
            except Exception:
                continue

            if score is None:
                continue

            result = {
                "lhs": tuple(lhs),
                "rhs": rhs,
                "score": score,
                "n_groups": n_groups,
                "violating_groups": violations
            }

            if score >= 0.99:
                strong_pair.append(result)
            elif score >= 0.95:
                moderate_pair.append(result)

    strong_pair = sorted(strong_pair, key=lambda x: (-x["score"], str(x["lhs"]), str(x["rhs"])))
    moderate_pair = sorted(moderate_pair, key=lambda x: (-x["score"], str(x["lhs"]), str(x["rhs"])))

    print("\nStrong pair-FD candidates (score >= 0.99):")
    for r in strong_pair[:30]:
        print(r)

    print("\nModerate pair-FD candidates (0.95 <= score < 0.99):")
    for r in moderate_pair[:30]:
        print(r)


############################################
# DATASET RUNNER
############################################

def run_dataset(name):
    print("\n" + "#" * 120)
    print("DATASET:", name)

    dataset = RDBDataset.from_relbench(name)
    rdb = dataset.rdb

    print("Tables:", rdb.tables.keys())

    for table_name, df in rdb.tables.items():
        try:
            analyze_table_basic(df, table_name)
            analyze_fd(df, table_name)
        except Exception as e:
            print(f"\nERROR while analyzing table {table_name}: {e}")
            traceback.print_exc()


############################################
# MAIN
############################################

if __name__ == "__main__":
    datasets = [
        "rel-amazon",
        "rel-hm",
        "rel-stack",
        "rel-trial"
    ]

    os.makedirs("logs", exist_ok=True)

    for ds in datasets:
        log_path = f"logs/{ds}.txt"
        print("\n" + "#" * 120)
        print(f"RUNNING DATASET: {ds}")
        print(f"Saving log to: {log_path}")

        original_stdout = sys.stdout

        try:
            with open(log_path, "w", encoding="utf-8") as f:
                sys.stdout = f
                try:
                    run_dataset(ds)
                except Exception as e:
                    print(f"\nERROR while running dataset {ds}: {e}")
                    traceback.print_exc()
        finally:
            sys.stdout = original_stdout

        print(f"Finished dataset: {ds}")
        print(f"Saved log to: {log_path}")