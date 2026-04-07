import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of value_counts with object-dtype is deprecated.*",
)

import argparse
import pandas as pd

from rdblearn.datasets import RDBDataset
import fastdfs.transform.type_transform as fastdfs_type_transform

from utils.common import set_global_seed
from patches.fastdfs_patch import patch_fastdfs_canonicalize_types
from pipeline.stage1_pruning import parse_extra_keep, run_stage1_pruning
from experiment.train_eval import run_one_experiment, choose_task


def print_table_pruning_summary(table_summary):
    if not table_summary:
        return

    print("\nTable pruning summary")
    print("-" * 60)

    items = []
    for table_name, stats in table_summary.items():
        items.append(f"{table_name}: {stats['before']} -> {stats['after']}")

    mid = (len(items) + 1) // 2
    left = items[:mid]
    right = items[mid:]

    max_left = max((len(x) for x in left), default=0)

    for i in range(max(len(left), len(right))):
        ltxt = left[i] if i < len(left) else ""
        rtxt = right[i] if i < len(right) else ""
        print(f"{ltxt:<{max_left + 4}}{rtxt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fd_thresholds", type=str, default="0.95,0.9,0.8")
    parser.add_argument("--near_unique_threshold", type=float, default=0.995)
    parser.add_argument("--extra_keep", type=str, default="")
    parser.add_argument("--list_tasks_only", action="store_true")
    parser.add_argument("--no_save_csv", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    patch_fastdfs_canonicalize_types(fastdfs_type_transform)

    dataset = RDBDataset.from_relbench(args.dataset)
    chosen_task_name, task = choose_task(dataset, args.task)

    if args.list_tasks_only:
        return

    thresholds = []
    for x in args.fd_thresholds.split(","):
        x = x.strip()
        if x:
            thresholds.append(float(x))

    if len(thresholds) == 0:
        raise ValueError("No valid thresholds found in --fd_thresholds")

    thresholds = sorted(thresholds, reverse=True)
    user_extra_keep = parse_extra_keep(args.extra_keep)

    baseline_result = run_one_experiment(
        tag="baseline",
        rdb=dataset.rdb,
        task=task,
        device=args.device,
        save_csv=(not args.no_save_csv),
    )

    results = [{
        "setting": "Baseline",
        "threshold": None,
        "metrics": baseline_result["metrics"],
        "table_summary": None,
    }]

    base_m = baseline_result["metrics"]
    summary_rows = []

    if base_m["task_type"] == "binary":
        summary_rows.append({
            "setting": "Baseline",
            "roc_auc": base_m["roc_auc"],
            "pr_auc": base_m["pr_auc"],
            "acc_0.5": base_m["acc_0.5"],
            "delta_roc_auc": 0.0,
            "delta_pr_auc": 0.0,
            "delta_acc_0.5": 0.0,
        })
    else:
        summary_rows.append({
            "setting": "Baseline",
            "rmse": base_m["rmse"],
            "mae": base_m["mae"],
            "r2": base_m["r2"],
            "delta_rmse": 0.0,
            "delta_mae": 0.0,
            "delta_r2": 0.0,
        })

    for th in thresholds:
        pruned_rdb, keep_columns, score_tables, table_pruning_summary = run_stage1_pruning(
            dataset_rdb=dataset.rdb,
            key_mappings=task.metadata.key_mappings,
            fd_threshold=th,
            near_unique_threshold=args.near_unique_threshold,
            extra_keep=user_extra_keep,
        )

        result = run_one_experiment(
            tag=f"stage1_fd_{str(th).replace('.', 'p')}",
            rdb=pruned_rdb,
            task=task,
            device=args.device,
            save_csv=(not args.no_save_csv),
        )

        results.append({
            "setting": f"FD ({th})",
            "threshold": th,
            "metrics": result["metrics"],
            "table_summary": table_pruning_summary,
        })

        m = result["metrics"]

        if base_m["task_type"] == "binary":
            summary_rows.append({
                "setting": f"FD ({th})",
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "acc_0.5": m["acc_0.5"],
                "delta_roc_auc": m["roc_auc"] - base_m["roc_auc"],
                "delta_pr_auc": m["pr_auc"] - base_m["pr_auc"],
                "delta_acc_0.5": m["acc_0.5"] - base_m["acc_0.5"],
            })
        else:
            summary_rows.append({
                "setting": f"FD ({th})",
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "delta_rmse": m["rmse"] - base_m["rmse"],
                "delta_mae": m["mae"] - base_m["mae"],
                "delta_r2": m["r2"] - base_m["r2"],
            })

        print(f"\nDataset   : {args.dataset}")
        print(f"Task      : {chosen_task_name}")
        print(f"Threshold : {th}")
        print_table_pruning_summary(table_pruning_summary)

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 80)
    print(f"Applied Pruning (Dataset: {args.dataset})")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    if not args.no_save_csv:
        summary_df.to_csv("comparison_all_thresholds.csv", index=False)


if __name__ == "__main__":
    main()