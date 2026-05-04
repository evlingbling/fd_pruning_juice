import argparse
import numpy as np
import pandas as pd

from rdblearn.datasets import RDBDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tabpfn import TabPFNClassifier

from pipeline.stage1_pruning_linear_gnn import (
    rdb_to_tables,
    build_fd_edges,
)


def extract_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    for attr in ["df", "data", "table", "_df", "_data"]:
        if hasattr(obj, attr):
            x = getattr(obj, attr)
            if isinstance(x, pd.DataFrame):
                return x.copy()

    if hasattr(obj, "to_pandas"):
        x = obj.to_pandas()
        if isinstance(x, pd.DataFrame):
            return x.copy()

    raise ValueError(f"Could not extract DataFrame from {type(obj)}")


def get_task_df(task, split):
    candidates = [
        f"{split}_table",
        f"{split}_df",
        split,
    ]

    for name in candidates:
        if hasattr(task, name):
            obj = getattr(task, name)
            if callable(obj):
                try:
                    return extract_df(obj())
                except TypeError:
                    pass
            else:
                return extract_df(obj)

    for method_name in [
        "get_table",
        "get_task_table",
        "get_split",
        "make_table",
    ]:
        if hasattr(task, method_name):
            method = getattr(task, method_name)
            try:
                return extract_df(method(split))
            except Exception:
                pass

    raise ValueError(
        f"Could not get split={split} from task. "
        f"Available attrs: {[x for x in dir(task) if 'table' in x.lower() or 'split' in x.lower() or split in x.lower()]}"
    )


def infer_label_col(df):
    candidates = ["target", "label", "churn", "y"]
    for c in candidates:
        if c in df.columns:
            return c

    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("Could not infer label column. Pass --label_col manually.")

    return numeric_cols[-1]


def infer_key_col(task_df, rdb_df):
    common = [c for c in task_df.columns if c in rdb_df.columns]
    common = [c for c in common if c.lower() not in {"target", "label", "churn", "y"}]

    id_like = [
        c for c in common
        if "id" in c.lower() or "customer" in c.lower() or "user" in c.lower()
    ]

    if id_like:
        return id_like[0], id_like[0]

    if common:
        return common[0], common[0]

    raise ValueError(
        "Could not infer key column. Pass --task_key_col and --rdb_key_col manually."
    )


def encode_series(s):
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce").astype(float)
        med = x.median()
        if np.isnan(med):
            med = 0.0
        x = x.fillna(med)

    elif pd.api.types.is_datetime64_any_dtype(s):
        x = s.astype("int64").astype(float)
        x = pd.Series(x, index=s.index)
        x = x.replace(-9223372036854775808, np.nan)
        med = x.median()
        if np.isnan(med):
            med = 0.0
        x = x.fillna(med)

    else:
        x = pd.Series(pd.factorize(s.astype("string"))[0], index=s.index).astype(float)

    std = x.std()
    if std == 0 or np.isnan(std):
        return x * 0.0

    return (x - x.mean()) / std


def build_hypergraph_only_features(
    df,
    fd_threshold=0.85,
    fd_source_mode="topk",
    fd_top_k=3,
    aggregation="mean",
):
    df = df.copy()
    cols = list(df.columns)

    if len(cols) == 0:
        raise ValueError("No columns left for hypergraph feature construction.")

    encoded = pd.DataFrame(index=df.index)
    for col in cols:
        encoded[col] = encode_series(df[col])

    edges, score_table, selected_fd = build_fd_edges(
        df=df,
        fd_threshold=fd_threshold,
        fd_source_mode=fd_source_mode,
        fd_top_k=fd_top_k,
    )

    incoming = {c: [] for c in cols}
    for lhs, rhs, w in edges:
        if lhs in encoded.columns and rhs in encoded.columns:
            incoming[rhs].append((lhs, float(w)))

    out = pd.DataFrame(index=df.index)

    for rhs in cols:
        msgs = []
        weights = []

        # self message
        msgs.append(encoded[rhs].to_numpy(dtype=float))
        weights.append(1.0)

        # FD-based incoming messages
        for lhs, w in incoming.get(rhs, []):
            msgs.append(encoded[lhs].to_numpy(dtype=float))
            weights.append(w)

        mat = np.vstack(msgs)
        weights = np.asarray(weights, dtype=float).reshape(-1, 1)

        if aggregation == "sum":
            agg = (mat * weights).sum(axis=0)
        elif aggregation == "mean":
            agg = (mat * weights).sum(axis=0) / max(weights.sum(), 1e-12)
        else:
            raise ValueError(f"Unsupported aggregation={aggregation}. Use mean or sum.")

        out[f"hg__{rhs}"] = agg

    print(f"[HG-ONLY] num input cols: {len(cols)}")
    print(f"[HG-ONLY] num FD edges: {len(edges)}")
    print(f"[HG-ONLY] num output features: {out.shape[1]}")

    return out


def sample_for_tabpfn(X, y, max_train_samples=50000, seed=42, stratify=True):
    n = len(X)

    if max_train_samples is None or max_train_samples <= 0 or n <= max_train_samples:
        print(f"[INFO] TabPFN training samples = {n} / {n} (no sampling)")
        return X, y

    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y)

    if stratify and len(np.unique(y_arr)) == 2:
        idx_parts = []

        for cls in np.unique(y_arr):
            cls_idx = np.where(y_arr == cls)[0]
            cls_frac = len(cls_idx) / n
            cls_take = int(round(max_train_samples * cls_frac))
            cls_take = max(1, min(cls_take, len(cls_idx)))

            sampled = rng.choice(cls_idx, size=cls_take, replace=False)
            idx_parts.append(sampled)

        idx = np.concatenate(idx_parts)

        if len(idx) > max_train_samples:
            idx = rng.choice(idx, size=max_train_samples, replace=False)

        if len(idx) < max_train_samples:
            remaining = np.setdiff1d(np.arange(n), idx, assume_unique=False)
            extra = rng.choice(
                remaining,
                size=min(max_train_samples - len(idx), len(remaining)),
                replace=False,
            )
            idx = np.concatenate([idx, extra])

        rng.shuffle(idx)

    else:
        idx = rng.choice(n, size=max_train_samples, replace=False)

    X_sampled = X.iloc[idx].reset_index(drop=True)
    y_sampled = y.iloc[idx].reset_index(drop=True)

    print(f"[INFO] TabPFN training samples = {len(X_sampled)} / {n}")
    return X_sampled, y_sampled


def evaluate(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "acc_0.5": accuracy_score(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--target_table", type=str, default="customer")

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="val")

    parser.add_argument("--task_key_col", type=str, default=None)
    parser.add_argument("--rdb_key_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)

    parser.add_argument("--fd_threshold", type=float, default=0.85)
    parser.add_argument("--fd_source_mode", type=str, default="topk")
    parser.add_argument("--fd_top_k", type=int, default=3)
    parser.add_argument("--aggregation", type=str, default="mean")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_samples", type=int, default=50000)
    parser.add_argument("--no_stratified_sample", action="store_true")
    parser.add_argument("--ignore_pretraining_limits", action="store_true")

    args = parser.parse_args()
    np.random.seed(args.seed)

    dataset = RDBDataset.from_relbench(args.dataset)
    task = dataset.tasks[args.task]

    tables = rdb_to_tables(dataset.rdb)

    if args.target_table not in tables:
        raise ValueError(
            f"target_table={args.target_table} not found. "
            f"Available tables: {list(tables.keys())}"
        )

    target_df = tables[args.target_table].copy()

    train_df = get_task_df(task, args.train_split)
    eval_df = get_task_df(task, args.eval_split)

    label_col = args.label_col or infer_label_col(train_df)

    task_key_col = args.task_key_col
    rdb_key_col = args.rdb_key_col

    if task_key_col is None or rdb_key_col is None:
        inferred_task_key, inferred_rdb_key = infer_key_col(train_df, target_df)
        task_key_col = task_key_col or inferred_task_key
        rdb_key_col = rdb_key_col or inferred_rdb_key

    print(f"[INFO] target_table = {args.target_table}")
    print(f"[INFO] task_key_col = {task_key_col}")
    print(f"[INFO] rdb_key_col  = {rdb_key_col}")
    print(f"[INFO] label_col    = {label_col}")

    # IMPORTANT:
    # key/label columns are used only for merging/evaluation.
    # They must NOT be used as hypergraph input features.
    exclude_cols = {rdb_key_col}

    if label_col in target_df.columns:
        exclude_cols.add(label_col)

    hg_input_df = target_df.drop(
        columns=[c for c in exclude_cols if c in target_df.columns]
    )

    print(f"[INFO] excluded from HG features: {sorted(exclude_cols)}")
    print(f"[INFO] HG input columns: {list(hg_input_df.columns)}")

    hg_features = build_hypergraph_only_features(
        df=hg_input_df,
        fd_threshold=args.fd_threshold,
        fd_source_mode=args.fd_source_mode,
        fd_top_k=args.fd_top_k,
        aggregation=args.aggregation,
    )

    # Attach key only for merging, not as a model feature.
    hg_features[rdb_key_col] = target_df[rdb_key_col].values

    train_merged = train_df[[task_key_col, label_col]].merge(
        hg_features,
        left_on=task_key_col,
        right_on=rdb_key_col,
        how="inner",
    )

    eval_merged = eval_df[[task_key_col, label_col]].merge(
        hg_features,
        left_on=task_key_col,
        right_on=rdb_key_col,
        how="inner",
    )

    feature_cols = [
        c for c in hg_features.columns
        if c != rdb_key_col
    ]

    X_train = train_merged[feature_cols].astype(float)
    y_train = train_merged[label_col].astype(int)

    X_eval = eval_merged[feature_cols].astype(float)
    y_eval = eval_merged[label_col].astype(int)

    print(f"[INFO] train_merged shape = {train_merged.shape}")
    print(f"[INFO] eval_merged shape  = {eval_merged.shape}")
    print(f"[INFO] X_train shape      = {X_train.shape}")
    print(f"[INFO] X_eval shape       = {X_eval.shape}")
    print(f"[INFO] feature cols       = {feature_cols}")

    X_train_fit, y_train_fit = sample_for_tabpfn(
        X_train,
        y_train,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
        stratify=not args.no_stratified_sample,
    )

    clf = TabPFNClassifier(
        device=args.device,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
    )

    clf.fit(X_train_fit, y_train_fit)

    y_prob = clf.predict_proba(X_eval)[:, 1]
    metrics = evaluate(y_eval.to_numpy(), y_prob)

    print("\n" + "=" * 80)
    print(f"HYPERGRAPH ONLY RESULT (seed {args.seed})")
    print("=" * 80)
    print(pd.DataFrame([{
        "setting": "HypergraphOnly",
        "train_samples_used": len(X_train_fit),
        "train_samples_total": len(X_train),
        **metrics,
    }]).to_string(index=False))


if __name__ == "__main__":
    main()