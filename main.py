import warnings
warnings.filterwarnings("ignore")

import argparse
import copy
import numpy as np
import pandas as pd
import fastdfs.transform.type_transform as fastdfs_type_transform

from patches.fastdfs_patch import patch_fastdfs_canonicalize_types
from rdblearn.datasets import RDBDataset
from experiment.train_eval import run_one_experiment

from pipeline.stage_1_GCN import (
    run_stage1_pruning_hypergraph,
    rebuild_rdb_metadata_from_data,
    coerce_rdb_key_columns_to_string,
)


def sanitize_rdb_for_fastdfs(rdb, source_rdb=None, key_mappings=None):
    rdb = rebuild_rdb_metadata_from_data(rdb)
    rdb = coerce_rdb_key_columns_to_string(rdb, key_mappings)
    rdb = rebuild_rdb_metadata_from_data(rdb)
    return rdb


def is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x)


def make_summary_rows(base_m, hg_m, hg_name):
    metric_keys = sorted(set(base_m.keys()) | set(hg_m.keys()))

    base_row = {"setting": "JUICE baseline"}
    hg_row = {"setting": hg_name}

    for k in metric_keys:
        base_val = base_m.get(k, np.nan)
        hg_val = hg_m.get(k, np.nan)

        base_row[k] = base_val
        hg_row[k] = hg_val

        if is_number(base_val) and is_number(hg_val):
            base_row[f"delta_{k}"] = 0.0
            hg_row[f"delta_{k}"] = float(hg_val) - float(base_val)

    return [base_row, hg_row]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fd_threshold", type=float, default=0.85)
    parser.add_argument("--fd_source_mode", type=str, default="topk")
    parser.add_argument("--fd_top_k", type=int, default=3)
    parser.add_argument("--min_hyperedge_size", type=int, default=2)

    parser.add_argument("--drop_score_threshold", type=float, default=0.8)

    parser.add_argument("--hg_hidden_dim", type=int, default=None)
    parser.add_argument("--hg_epochs", type=int, default=100)
    parser.add_argument("--hg_lr", type=float, default=1e-2)
    parser.add_argument("--hg_weight_decay", type=float, default=1e-4)
    parser.add_argument("--hg_dropout", type=float, default=0.0)

    args = parser.parse_args()
    np.random.seed(args.seed)

    patch_fastdfs_canonicalize_types(fastdfs_type_transform)

    dataset = RDBDataset.from_relbench(args.dataset)
    task = dataset.tasks[args.task]

    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIG")
    print("=" * 80)
    print(f"Dataset              : {args.dataset}")
    print(f"Task                 : {args.task}")
    print(f"Seed                 : {args.seed}")
    print(f"Device               : {args.device}")
    print(f"FD threshold         : {args.fd_threshold}")
    print(f"FD source mode       : {args.fd_source_mode}")
    print(f"FD top-k             : {args.fd_top_k}")
    print(f"Min hyperedge size   : {args.min_hyperedge_size}")
    print(f"Drop score threshold : {args.drop_score_threshold}")
    print(f"HG hidden dim        : {args.hg_hidden_dim}")
    print(f"HG epochs            : {args.hg_epochs}")
    print(f"HG lr                : {args.hg_lr}")
    print(f"HG weight decay      : {args.hg_weight_decay}")
    print(f"HG dropout           : {args.hg_dropout}")

    baseline_rdb = copy.deepcopy(dataset.rdb)
    baseline_rdb = sanitize_rdb_for_fastdfs(
        baseline_rdb,
        dataset.rdb,
        task.metadata.key_mappings,
    )

    baseline_result = run_one_experiment(
        tag="baseline",
        rdb=baseline_rdb,
        task=task,
        device=args.device,
        save_csv=False,
        dfs_enabled=True,
    )

    base_m = baseline_result["metrics"]

    print("\nBaseline metrics")
    print("-" * 60)
    print(base_m)

    pruned_rdb, keep_columns, score_tables, table_pruning_summary, hg_debug = run_stage1_pruning_hypergraph(
        dataset_rdb=dataset.rdb,
        key_mappings=task.metadata.key_mappings,
        fd_threshold=args.fd_threshold,
        fd_source_mode=args.fd_source_mode,
        fd_top_k=args.fd_top_k,
        min_hyperedge_size=args.min_hyperedge_size,
        drop_score_threshold=args.drop_score_threshold,
        hg_hidden_dim=args.hg_hidden_dim,
        hg_epochs=args.hg_epochs,
        hg_lr=args.hg_lr,
        hg_weight_decay=args.hg_weight_decay,
        hg_dropout=args.hg_dropout,
        device=args.device,
    )

    print("\nTable pruning summary")
    print("-" * 60)
    for table_name, s in table_pruning_summary.items():
        print(f"{table_name}: {s['before']} -> {s['after']}")

    print("\nHypergraphConv debug preview")
    print("-" * 60)
    for table_name, dbg in hg_debug.items():
        hyperedges = dbg.get("hyperedges", [])
        selected_fd = dbg.get("selected_fd", pd.DataFrame())
        print(
            f"{table_name}: "
            f"{len(hyperedges)} hyperedges, "
            f"{len(selected_fd)} selected FDs"
        )

    pruned_rdb = sanitize_rdb_for_fastdfs(
        pruned_rdb,
        dataset.rdb,
        task.metadata.key_mappings,
    )

    hg_result = run_one_experiment(
        tag="hypergraph_conv_pruning",
        rdb=pruned_rdb,
        task=task,
        device=args.device,
        save_csv=False,
        dfs_enabled=True,
    )

    hg_m = hg_result["metrics"]

    print("\nHypergraphConv metrics")
    print("-" * 60)
    print(hg_m)

    summary_rows = make_summary_rows(
        base_m=base_m,
        hg_m=hg_m,
        hg_name="JUICE + PyG HypergraphConv pruning",
    )

    df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()