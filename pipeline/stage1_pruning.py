# STAGE 1 PRUNING MODULE

from typing import Dict, List, Set, Optional, Tuple
import numpy as np
import pandas as pd

from utils.common import plain_df_copy

from fd.heuristics import (
    is_identifier_or_time,
    is_low_information_rhs,
    get_fd_drop_tier,
    is_near_unique,
    is_probably_measure_col,
    is_probably_text_col,
    dominant_ratio,
    is_time_like,
    is_id_like,
)

from fd.scoring import compute_best_fd_per_rhs

def extract_table_df(table):
    if isinstance(table, pd.DataFrame):
        return plain_df_copy(table)

    for attr in ["df", "data", "table", "_df", "_data"]:
        if hasattr(table, attr):
            obj = getattr(table, attr)
            if isinstance(obj, pd.DataFrame):
                return plain_df_copy(obj)

    if hasattr(table, "to_pandas"):
        obj = table.to_pandas()
        if isinstance(obj, pd.DataFrame):
            return plain_df_copy(obj)
        
    raise AttributeError


def rdb_to_tables(rdb) -> Dict[str, pd.DataFrame]:
    return {name: extract_table_df(table) for name, table in rdb.tables.items()}


def parse_key_mappings(key_mappings) -> Dict[str, List[str]]:
    always_keep = {}
    if key_mappings is None:
        return always_keep

    for _, ref in key_mappings.items():
        if isinstance(ref, str) and "." in ref:
            t, c = ref.split(".", 1)
            always_keep.setdefault(t, []).append(c)

    return {k: sorted(set(v)) for k, v in always_keep.items()}

def parse_extra_keep(extra_keep_str: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    if not extra_keep_str:
        return result

    for chunk in extra_keep_str.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue

        table_name, cols_str = chunk.split(":", 1)
        table_name = table_name.strip()
        cols = [c.strip() for c in cols_str.split(",") if c.strip()]

        if table_name and cols:
            result.setdefault(table_name, [])
            result[table_name].extend(cols)

    return {k: sorted(set(v)) for k, v in result.items()}


def merge_keep_dicts(*dicts):
    merged = {}
    for d in dicts:
        if not d:
            continue
        for t, cols in d.items():
            merged.setdefault(t, []).extend(cols)

    return {k: sorted(set(v)) for k, v in merged.items()}



# SELECT TABLES

def guess_prune_tables(tables, key_mappings=None) -> Set[str]:
    always_keep = parse_key_mappings(key_mappings)
    prune_tables = set()

    for t, df in tables.items():
        if len(df.columns) < 3:
            continue

        has_key = any(is_identifier_or_time(c) for c in df.columns) or (t in always_keep)

        if has_key or len(df.columns) >= 4:
            prune_tables.add(t)

    return prune_tables


# CORE PRUNING

def build_keep_columns_schema_safe(
    tables,
    key_mappings,
    fd_threshold=0.95,
    prune_tables=None,
    extra_keep=None,
    near_unique_threshold=0.995,
):
    if prune_tables is None:
        prune_tables = set()

    always_keep = parse_key_mappings(key_mappings)
    if extra_keep:
        always_keep = merge_keep_dicts(always_keep, extra_keep)

    keep_columns = {}
    score_tables = {}

    for table_name, df in tables.items():
        cols = list(df.columns)

        if table_name not in prune_tables:
            keep_columns[table_name] = cols
            continue

        base_keep = set(always_keep.get(table_name, []))
        base_keep |= {c for c in cols if is_identifier_or_time(c)}

        best_fd = compute_best_fd_per_rhs(df)
        score_tables[table_name] = best_fd

        drop = set()

        for _, row in best_fd.iterrows():
            rhs, lhs, score = row["rhs"], row["best_lhs"], row["best_score"]

            if rhs in base_keep or score < fd_threshold:
                continue

            if rhs not in df.columns or lhs not in df.columns:
                continue

            drop_tier = get_fd_drop_tier(rhs, df[rhs])
            rhs_low = is_low_information_rhs(rhs, df[rhs])

            if drop_tier == "keep" and not rhs_low:
                continue

            try:
                lhs_unique = is_near_unique(df[lhs], threshold=near_unique_threshold)
            except Exception:
                lhs_unique = False

            lhs_key = lhs in base_keep or is_identifier_or_time(lhs) or lhs_unique

            if drop_tier == "hard_drop":
                drop.add(rhs)
                continue

            if rhs_low and lhs_key:
                drop.add(rhs)
                continue

            if drop_tier == "soft_drop" and lhs_key:
                drop.add(rhs)

        keep_columns[table_name] = [c for c in cols if c not in drop]

    return keep_columns, score_tables

def make_pruned_rdb(rdb, keep_columns):
    from copy import deepcopy

    pruned = deepcopy(rdb)

    for t, df in pruned.tables.items():
        keep = keep_columns.get(t, df.columns)
        new_df = plain_df_copy(df.loc[:, keep])

        for col in new_df.columns:
            if is_time_like(col):
                try:
                    new_df[col] = pd.to_datetime(new_df[col], errors="coerce")
                except Exception:
                    pass

        pruned.tables[t] = new_df
    pruned = rebuild_rdb_metadata_from_data(pruned)
    return pruned

def _coerce_key_series_to_string(s):
    return s.astype("string")


def coerce_rdb_key_columns_to_string(rdb, key_mappings=None):
    key_cols = {}

    for _, ref in (key_mappings or {}).items():
        if "." in ref:
            t, c = ref.split(".", 1)
            key_cols.setdefault(t, set()).add(c)

    for t, df in rdb.tables.items():
        for col in df.columns:
            if is_id_like(col):
                key_cols.setdefault(t, set()).add(col)

    for t, cols in key_cols.items():
        if t not in rdb.tables:
            continue
        df = plain_df_copy(rdb.tables[t])
        for col in cols:
            if col in df.columns:
                df[col] = _coerce_key_series_to_string(df[col])
        rdb.tables[t] = df
    
    rdb = rebuild_rdb_metadata_from_data(rdb)
    return rdb

def rebuild_rdb_metadata_from_data(rdb):
    if not hasattr(rdb, "metadata") or not hasattr(rdb.metadata, "tables"):
        return rdb

    for table_name, df in rdb.tables.items():
        if table_name not in rdb.metadata.tables:
            continue

        actual_cols = set(df.columns)
        meta = rdb.metadata.tables[table_name]

        def filter_value(value):
            if isinstance(value, dict):
                return {
                    k: v for k, v in value.items()
                    if (not isinstance(k, str)) or (k in actual_cols)
                }

            if isinstance(value, list):
                filtered = []
                for x in value:
                    if hasattr(x, "name"):
                        if x.name in actual_cols:
                            filtered.append(x)
                    elif isinstance(x, str):
                        if x in actual_cols:
                            filtered.append(x)
                    else:
                        filtered.append(x)
                return filtered

            if isinstance(value, tuple):
                filtered = []
                for x in value:
                    if hasattr(x, "name"):
                        if x.name in actual_cols:
                            filtered.append(x)
                    elif isinstance(x, str):
                        if x in actual_cols:
                            filtered.append(x)
                    else:
                        filtered.append(x)
                return tuple(filtered)

            if isinstance(value, set):
                filtered = set()
                for x in value:
                    if hasattr(x, "name"):
                        if x.name in actual_cols:
                            filtered.add(x)
                    elif isinstance(x, str):
                        if x in actual_cols:
                            filtered.add(x)
                    else:
                        try:
                            filtered.add(x)
                        except Exception:
                            pass
                return filtered

            return value

        if hasattr(meta, "columns"):
            try:
                new_cols = []
                for col in meta.columns:
                    if hasattr(col, "name"):
                        if col.name in actual_cols:
                            new_cols.append(col)
                    elif isinstance(col, str):
                        if col in actual_cols:
                            new_cols.append(col)
                    else:
                        new_cols.append(col)
                meta.columns = new_cols
            except Exception:
                pass

        for attr in [
            "column_names",
            "column_types",
            "logical_types",
            "semantic_tags",
            "dtypes",
            "dtype_map",
            "logical_type_map",
            "data_types",
            "feature_types",
            "col_types",
            "schema",
            "column_metadata",
            "column_stats",
            "ww_schema",
        ]:
            if hasattr(meta, attr):
                try:
                    setattr(meta, attr, filter_value(getattr(meta, attr)))
                except Exception:
                    pass

        if hasattr(meta, "__dict__"):
            for key, value in list(meta.__dict__.items()):
                try:
                    if key == "columns":
                        new_cols = []
                        for col in value:
                            if hasattr(col, "name"):
                                if col.name in actual_cols:
                                    new_cols.append(col)
                            elif isinstance(col, str):
                                if col in actual_cols:
                                    new_cols.append(col)
                            else:
                                new_cols.append(col)
                        meta.__dict__[key] = new_cols
                    else:
                        meta.__dict__[key] = filter_value(value)
                except Exception:
                    pass

    return rdb



# RUNNER

def run_stage1_pruning(
    dataset_rdb,
    key_mappings,
    fd_threshold=0.95,
    near_unique_threshold=0.995,
    extra_keep=None,
):
    tables = rdb_to_tables(dataset_rdb)
    prune_tables = guess_prune_tables(tables, key_mappings)

    keep_columns, score_tables = build_keep_columns_schema_safe(
        tables,
        key_mappings,
        fd_threshold,
        prune_tables,
        extra_keep,
        near_unique_threshold,
    )

    table_pruning_summary = {}
    for table_name, df in tables.items():
        before_ncols = len(df.columns)
        after_ncols = len(keep_columns.get(table_name, list(df.columns)))
        table_pruning_summary[table_name] = {
            "before": before_ncols,
            "after": after_ncols,
        }

    pruned_rdb = make_pruned_rdb(dataset_rdb, keep_columns)
    pruned_rdb = coerce_rdb_key_columns_to_string(pruned_rdb, key_mappings)
    pruned_rdb = rebuild_rdb_metadata_from_data(pruned_rdb)

    return pruned_rdb, keep_columns, score_tables, table_pruning_summary