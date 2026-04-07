import random
import numpy as np
import pandas as pd


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def plain_df_copy(df: pd.DataFrame) -> pd.DataFrame:
    try:
        out = pd.DataFrame(df.copy(deep=True))
        out.index = df.index.copy()
        out.columns = df.columns.copy()
        return out
    except Exception:
        try:
            out = pd.DataFrame(df.to_dict(orient="list"), index=df.index.copy())
            out.columns = df.columns.copy()
            return out
        except Exception:
            return pd.DataFrame(
                np.asarray(df),
                index=df.index.copy(),
                columns=df.columns.copy(),
            )