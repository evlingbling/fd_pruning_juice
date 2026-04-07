from typing import Dict, Optional

import numpy as np
import pandas as pd

from rdblearn.estimator import RDBLearnRegressor
from tabpfn import TabPFNRegressor

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def infer_task_type(y: pd.Series) -> str:
    uniq = set(pd.Series(y).dropna().unique().tolist())
    if uniq.issubset({0, 1}):
        return "binary"
    return "regression"


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    task_type = infer_task_type(pd.Series(y_true))
    out = {"task_type": task_type}

    if task_type == "binary":
        out["roc_auc"] = roc_auc_score(y_true, y_pred)
        out["pr_auc"] = average_precision_score(y_true, y_pred)
        out["acc_0.5"] = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    else:
        out["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        out["mae"] = mean_absolute_error(y_true, y_pred)
        out["r2"] = r2_score(y_true, y_pred)

    return out


def build_regressor(device: str = "cuda") -> RDBLearnRegressor:
    return RDBLearnRegressor(
        base_estimator=TabPFNRegressor(device=device),
        config={
            "dfs": {"max_depth": 2},
            "enable_target_augmentation": True,
            "temporal_diff": {"enabled": True},
            "max_train_samples": 1000,
        },
    )


def run_one_experiment(
    tag: str,
    rdb,
    task,
    device: str = "cuda",
    save_csv: bool = False,
):
    reg = build_regressor(device=device)

    X_train = task.train_df.drop(columns=[task.metadata.target_col])
    y_train = task.train_df[task.metadata.target_col]

    X_test = task.test_df.drop(columns=[task.metadata.target_col])
    y_test = task.test_df[task.metadata.target_col]

    reg.fit(
        X=X_train,
        y=y_train,
        rdb=rdb,
        key_mappings=task.metadata.key_mappings,
        cutoff_time_column=task.metadata.time_col,
    )

    predictions = reg.predict(X=X_test)

    df_pred = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": predictions,
        }
    )

    if save_csv:
        df_pred.to_csv(f"predictions_{tag}.csv", index=False)

    metrics = evaluate_predictions(y_test, predictions)

    return {
        "tag": tag,
        "metrics": metrics,
        "pred_df": df_pred,
        "reg": reg,
    }


def choose_task(dataset, task_name: Optional[str]):
    available = list(dataset.tasks.keys())

    if len(available) == 0:
        raise ValueError("No tasks found in dataset.")

    if task_name is None:
        chosen = available[0]
        return chosen, dataset.tasks[chosen]

    if task_name not in dataset.tasks:
        raise ValueError(
            f"Task '{task_name}' not found. Available tasks: {available}"
        )

    return task_name, dataset.tasks[task_name]