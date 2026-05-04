from typing import Dict, List, Set
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

from utils.common import plain_df_copy

from fd.heuristics import (
    is_identifier_or_time,
    is_low_information_rhs,
    get_fd_drop_tier,
    is_near_unique,
    dominant_ratio,
    is_time_like,
    is_id_like,
)

from fd.scoring import compute_best_fd_per_rhs

def is_structural_column(col) -> bool:
    col_l = str(col).lower()
    protected_keywords = [
        "id",
        "key",
        "index",
        "group",
        "time",
        "date",
    ]
    return any(k in col_l for k in protected_keywords)


def get_structural_keep_columns(df: pd.DataFrame) -> Set[str]:
    return {c for c in df.columns if is_structural_column(c) or is_identifier_or_time(c)}


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

    raise AttributeError("Could not extract pandas DataFrame from table object.")


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


def guess_prune_tables(tables, key_mappings=None) -> Set[str]:
    always_keep = parse_key_mappings(key_mappings)
    prune_tables = set()

    for t, df in tables.items():
        if len(df.columns) < 3:
            continue

        has_key = any(is_structural_column(c) for c in df.columns) or (t in always_keep)

        if has_key or len(df.columns) >= 4:
            prune_tables.add(t)

    return prune_tables


def make_pruned_rdb(rdb, keep_columns):
    from copy import deepcopy

    pruned = deepcopy(rdb)

    for t, df in pruned.tables.items():
        keep = keep_columns.get(t, list(df.columns))

        keep_set = set(keep)
        keep_set |= get_structural_keep_columns(df)

        keep = [c for c in df.columns if c in keep_set]
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
            if is_id_like(col) or is_structural_column(col):
                key_cols.setdefault(t, set()).add(col)

    for t, cols in key_cols.items():
        if t not in rdb.tables:
            continue

        df = plain_df_copy(rdb.tables[t])
        for col in cols:
            if col in df.columns:
                try:
                    df[col] = _coerce_key_series_to_string(df[col])
                except Exception:
                    pass
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

def safe_entropy(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) == 0:
        return 0.0

    probs = s.astype("string").value_counts(normalize=True, dropna=True).values
    if len(probs) == 0:
        return 0.0

    return float(-(probs * np.log(probs + 1e-12)).sum())


def safe_distinct_ratio(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    return float(s.nunique(dropna=True) / max(len(s), 1))


def safe_missing_ratio(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    return float(s.isna().mean())


def build_column_features(
    df: pd.DataFrame,
    near_unique_threshold: float = 0.995,
) -> Dict[str, np.ndarray]:
    feats = {}

    for col in df.columns:
        s = df[col]

        try:
            near_unique = float(is_near_unique(s, threshold=near_unique_threshold))
        except Exception:
            near_unique = 0.0

        try:
            low_info = float(is_low_information_rhs(col, s))
        except Exception:
            low_info = 0.0

        try:
            dom = float(dominant_ratio(s))
        except Exception:
            dom = 0.0

        feat = np.array([
            safe_distinct_ratio(s),
            safe_missing_ratio(s),
            safe_entropy(s),
            dom,
            float(is_identifier_or_time(col) or is_structural_column(col)),
            float(is_id_like(col) or is_structural_column(col)),
            float(is_time_like(col) or ("date" in str(col).lower())),
            near_unique,
            low_info,
        ], dtype=float)

        feats[col] = feat

    return feats

def compute_unary_fd_score(lhs_s: pd.Series, rhs_s: pd.Series) -> float:
    tmp = pd.DataFrame({"lhs": lhs_s, "rhs": rhs_s}).dropna()
    if len(tmp) == 0:
        return 0.0

    total = len(tmp)
    weighted_purity = 0.0

    for _, g in tmp.groupby("lhs", dropna=True):
        if len(g) == 0:
            continue

        max_prob = g["rhs"].astype("string").value_counts(
            normalize=True,
            dropna=True,
        ).max()

        weighted_purity += (len(g) / total) * float(max_prob)

    return float(weighted_purity)


def compute_all_unary_fd_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = list(df.columns)

    for lhs in cols:
        for rhs in cols:
            if lhs == rhs:
                continue

            try:
                score = compute_unary_fd_score(df[lhs], df[rhs])
            except Exception:
                score = 0.0

            rows.append({
                "lhs": lhs,
                "rhs": rhs,
                "score": float(score),
            })

    if not rows:
        return pd.DataFrame(columns=["lhs", "rhs", "score"])

    return pd.DataFrame(rows).sort_values(
        ["rhs", "score", "lhs"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def select_fd_sources(
    df: pd.DataFrame,
    fd_threshold: float = 0.95,
    fd_source_mode: str = "best",
    fd_top_k: int = 2,
):
    if fd_source_mode not in {"best", "topk", "threshold_all"}:
        raise ValueError(
            f"Invalid fd_source_mode={fd_source_mode}. "
            f"Choose from ['best', 'topk', 'threshold_all']"
        )

    if fd_source_mode == "best":
        best_fd = compute_best_fd_per_rhs(df)

        selected = []
        for _, row in best_fd.iterrows():
            lhs = row["best_lhs"]
            rhs = row["rhs"]
            score = float(row["best_score"])

            if lhs not in df.columns or rhs not in df.columns:
                continue
            if lhs == rhs:
                continue
            if score < fd_threshold:
                continue

            selected.append({
                "lhs": lhs,
                "rhs": rhs,
                "score": score,
            })

        selected_df = pd.DataFrame(selected, columns=["lhs", "rhs", "score"])
        score_table = best_fd.copy()
        return selected_df, score_table

    all_fd = compute_all_unary_fd_scores(df)
    if len(all_fd) == 0:
        empty = pd.DataFrame(columns=["lhs", "rhs", "score"])
        return empty, empty

    all_fd = all_fd[all_fd["score"] >= fd_threshold].copy()

    if fd_source_mode == "threshold_all":
        selected_df = all_fd.sort_values(
            ["rhs", "score", "lhs"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        return selected_df, all_fd

    pieces = []
    for rhs, g in all_fd.groupby("rhs", sort=False):
        g = g.sort_values(["score", "lhs"], ascending=[False, True]).head(fd_top_k)
        pieces.append(g)

    if pieces:
        selected_df = pd.concat(pieces, axis=0).reset_index(drop=True)
    else:
        selected_df = pd.DataFrame(columns=["lhs", "rhs", "score"])

    return selected_df, all_fd



def build_hyperedges_from_fd(
    selected_fd: pd.DataFrame,
    min_hyperedge_size: int = 2,
):

    hyperedges = []

    if selected_fd is None or len(selected_fd) == 0:
        return hyperedges

    for rhs, group in selected_fd.groupby("rhs", sort=False):
        lhs_members = list(group["lhs"].dropna().astype(str).unique())
        rhs = str(rhs)

        members = lhs_members + [rhs]
        members = list(dict.fromkeys(members))

        if len(members) >= min_hyperedge_size:
            weight = float(group["score"].mean())
            hyperedges.append((members, weight))

    return hyperedges


def build_hyperedge_index(
    cols: List[str],
    hyperedges,
    device="cpu",
):

    col_to_idx = {c: i for i, c in enumerate(cols)}

    node_idx = []
    edge_idx = []
    hyperedge_weight = []

    valid_eid = 0

    for members, weight in hyperedges:
        valid_members = [m for m in members if m in col_to_idx]

        if len(valid_members) == 0:
            continue

        for m in valid_members:
            node_idx.append(col_to_idx[m])
            edge_idx.append(valid_eid)

        hyperedge_weight.append(float(weight))
        valid_eid += 1

    if len(node_idx) == 0:
        # fallback: one self-style hyperedge per node
        for i in range(len(cols)):
            node_idx.append(i)
            edge_idx.append(i)
            hyperedge_weight.append(1.0)

    hyperedge_index = torch.tensor(
        [node_idx, edge_idx],
        dtype=torch.long,
        device=device,
    )

    hyperedge_weight = torch.tensor(
        hyperedge_weight,
        dtype=torch.float32,
        device=device,
    )

    return hyperedge_index, hyperedge_weight

class FDHypergraphConvBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = None,
        out_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim

        self.conv1 = HypergraphConv(in_dim, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        x = self.conv1(
            x,
            hyperedge_index,
            hyperedge_weight=hyperedge_weight,
        )
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(
            x,
            hyperedge_index,
            hyperedge_weight=hyperedge_weight,
        )

        return x


def aggregate_fd_messages_hypergraph_conv(
    cols: List[str],
    node_features: Dict[str, np.ndarray],
    selected_fd: pd.DataFrame,
    hidden_dim: int = None,
    num_epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    min_hyperedge_size: int = 2,
    dropout: float = 0.0,
):

    if not node_features:
        return {}, []

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    x_np = np.stack([node_features[c] for c in cols], axis=0)
    x = torch.tensor(x_np, dtype=torch.float32, device=device)

    hyperedges = build_hyperedges_from_fd(
        selected_fd=selected_fd,
        min_hyperedge_size=min_hyperedge_size,
    )

    hyperedge_index, hyperedge_weight = build_hyperedge_index(
        cols=cols,
        hyperedges=hyperedges,
        device=device,
    )

    in_dim = x.shape[1]

    model = FDHypergraphConvBackbone(
        in_dim=in_dim,
        hidden_dim=hidden_dim or in_dim,
        out_dim=in_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()

        z = model(
            x=x,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
        )

        loss = F.mse_loss(z, x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model(
            x=x,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
        )

    z = z.detach().cpu().numpy()

    agg_features = {
        col: z[i]
        for i, col in enumerate(cols)
    }

    return agg_features, hyperedges

def compute_structural_drop_score(
    original_feat: np.ndarray,
    agg_feat: np.ndarray,
) -> float:
    distinct_ratio = original_feat[0]
    missing_ratio = original_feat[1]
    entropy = original_feat[2]
    is_keyish = max(original_feat[4], original_feat[5], original_feat[7])
    is_time = original_feat[6]
    low_info = original_feat[8]

    agg_distinct = agg_feat[0]
    agg_entropy = agg_feat[2]
    agg_dom = agg_feat[3]
    agg_low_info = agg_feat[8]

    score = 0.0

    score += 1.5 * low_info
    score += 1.0 * agg_low_info
    score += 0.7 * agg_dom
    score += 0.4 * missing_ratio

    score -= 1.5 * is_keyish
    score -= 1.2 * is_time
    score -= 0.5 * distinct_ratio
    score -= 0.3 * entropy
    score -= 0.3 * agg_distinct
    score -= 0.2 * agg_entropy

    return float(score)


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
        base_keep |= get_structural_keep_columns(df)

        best_fd = compute_best_fd_per_rhs(df)
        score_tables[table_name] = best_fd

        drop = set()

        for _, row in best_fd.iterrows():
            rhs, lhs, score = row["rhs"], row["best_lhs"], row["best_score"]

            if rhs in base_keep or is_structural_column(rhs) or score < fd_threshold:
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

            lhs_key = (
                lhs in base_keep
                or is_structural_column(lhs)
                or is_identifier_or_time(lhs)
                or lhs_unique
            )

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

def build_keep_columns_schema_safe_hypergraph_conv(
    tables,
    key_mappings,
    fd_threshold=0.95,
    prune_tables=None,
    extra_keep=None,
    near_unique_threshold=0.995,
    drop_score_threshold=0.8,
    fd_source_mode="topk",
    fd_top_k=3,
    min_hyperedge_size=2,
    hg_hidden_dim=None,
    hg_epochs=100,
    hg_lr=1e-2,
    hg_weight_decay=1e-4,
    hg_dropout=0.0,
    device="cpu",
):
    if prune_tables is None:
        prune_tables = set()

    always_keep = parse_key_mappings(key_mappings)
    if extra_keep:
        always_keep = merge_keep_dicts(always_keep, extra_keep)

    keep_columns = {}
    score_tables = {}
    hg_debug = {}

    for table_name, df in tables.items():
        cols = list(df.columns)

        if table_name not in prune_tables:
            keep_columns[table_name] = cols
            continue

        base_keep = set(always_keep.get(table_name, []))
        base_keep |= get_structural_keep_columns(df)

        node_features = build_column_features(
            df,
            near_unique_threshold=near_unique_threshold,
        )

        selected_fd, score_table = select_fd_sources(
            df=df,
            fd_threshold=fd_threshold,
            fd_source_mode=fd_source_mode,
            fd_top_k=fd_top_k,
        )

        agg_features, hyperedges = aggregate_fd_messages_hypergraph_conv(
            cols=cols,
            node_features=node_features,
            selected_fd=selected_fd,
            hidden_dim=hg_hidden_dim,
            num_epochs=hg_epochs,
            lr=hg_lr,
            weight_decay=hg_weight_decay,
            device=device,
            min_hyperedge_size=min_hyperedge_size,
            dropout=hg_dropout,
        )

        score_tables[table_name] = score_table

        drop = set()
        per_col_scores = {}

        for col in cols:
            if col in base_keep or is_structural_column(col):
                per_col_scores[col] = -999.0
                continue

            drop_tier = get_fd_drop_tier(col, df[col])
            rhs_low = is_low_information_rhs(col, df[col])

            if drop_tier == "keep" and not rhs_low:
                per_col_scores[col] = -999.0
                continue

            score = compute_structural_drop_score(
                original_feat=node_features[col],
                agg_feat=agg_features[col],
            )

            per_col_scores[col] = score

            if drop_tier == "hard_drop" and score >= 0.0:
                drop.add(col)
                continue

            if (rhs_low or drop_tier == "soft_drop") and score >= drop_score_threshold:
                drop.add(col)

        keep_columns[table_name] = [c for c in cols if c not in drop]

        hg_debug[table_name] = {
            "hyperedges": hyperedges,
            "selected_fd": selected_fd,
            "drop_scores": per_col_scores,
            "fd_source_mode": fd_source_mode,
            "fd_top_k": fd_top_k,
            "min_hyperedge_size": min_hyperedge_size,
            "hg_epochs": hg_epochs,
            "hg_lr": hg_lr,
            "hg_hidden_dim": hg_hidden_dim,
            "device": device,
        }

    return keep_columns, score_tables, hg_debug


def run_stage1_pruning_hypergraph(
    dataset_rdb,
    key_mappings,
    fd_threshold=0.95,
    near_unique_threshold=0.995,
    extra_keep=None,
    drop_score_threshold=0.8,
    min_hyperedge_size=2,
    fd_source_mode="topk",
    fd_top_k=3,
    hg_hidden_dim=None,
    hg_epochs=100,
    hg_lr=1e-2,
    hg_weight_decay=1e-4,
    hg_dropout=0.0,
    device="cpu",
):

    tables = rdb_to_tables(dataset_rdb)
    prune_tables = guess_prune_tables(tables, key_mappings)

    keep_columns, score_tables, hg_debug = build_keep_columns_schema_safe_hypergraph_conv(
        tables=tables,
        key_mappings=key_mappings,
        fd_threshold=fd_threshold,
        prune_tables=prune_tables,
        extra_keep=extra_keep,
        near_unique_threshold=near_unique_threshold,
        drop_score_threshold=drop_score_threshold,
        fd_source_mode=fd_source_mode,
        fd_top_k=fd_top_k,
        min_hyperedge_size=min_hyperedge_size,
        hg_hidden_dim=hg_hidden_dim,
        hg_epochs=hg_epochs,
        hg_lr=hg_lr,
        hg_weight_decay=hg_weight_decay,
        hg_dropout=hg_dropout,
        device=device,
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

    return pruned_rdb, keep_columns, score_tables, table_pruning_summary, hg_debug


def run_stage1_pruning_linear_gnn(
    dataset_rdb,
    key_mappings,
    fd_threshold=0.95,
    near_unique_threshold=0.995,
    extra_keep=None,
    aggregation="mean",
    drop_score_threshold=0.8,
    fd_source_mode="topk",
    fd_top_k=3,
    min_hyperedge_size=2,
    hg_hidden_dim=None,
    hg_epochs=100,
    hg_lr=1e-2,
    hg_weight_decay=1e-4,
    hg_dropout=0.0,
    device="cpu",
):
    return run_stage1_pruning_hypergraph(
        dataset_rdb=dataset_rdb,
        key_mappings=key_mappings,
        fd_threshold=fd_threshold,
        near_unique_threshold=near_unique_threshold,
        extra_keep=extra_keep,
        drop_score_threshold=drop_score_threshold,
        min_hyperedge_size=min_hyperedge_size,
        fd_source_mode=fd_source_mode,
        fd_top_k=fd_top_k,
        hg_hidden_dim=hg_hidden_dim,
        hg_epochs=hg_epochs,
        hg_lr=hg_lr,
        hg_weight_decay=hg_weight_decay,
        hg_dropout=hg_dropout,
        device=device,
    )