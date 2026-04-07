import pandas as pd


def is_id_like(col: str) -> bool:
    c = col.lower()
    return c == "id" or c.endswith("id") or c.endswith("_id")


def is_time_like(col: str) -> bool:
    c = col.lower()
    return ("date" in c) or ("time" in c) or ("timestamp" in c)


def is_identifier_or_time(col: str) -> bool:
    return is_id_like(col) or is_time_like(col)


def is_probably_measure_col(col: str) -> bool:
    c = col.lower()
    keywords = [ "price", "ctr", "click", "count", "num", "score", "rate", "hist", "position", "prob", "amount", "target", "label", "rank", "freq", "duration", "timeon", "value", "age"]
    return any(k in c for k in keywords)


def is_probably_text_col(series: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    sample = non_null.astype(str).head(200)
    avg_len = sample.str.len().mean()
    nunique = sample.nunique(dropna=False)

    return avg_len >= 15 or nunique >= max(20, int(len(sample) * 0.5))


def is_near_unique(series: pd.Series, threshold: float = 0.99) -> bool:
    if len(series) == 0:
        return False
    return (series.nunique(dropna=False) / len(series)) >= threshold


def dominant_ratio(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) == 0:
        return 1.0
    vc = s.value_counts(dropna=False, normalize=True)
    if len(vc) == 0:
        return 1.0
    return float(vc.iloc[0])


def is_low_cardinality(series: pd.Series, max_unique: int = 12) -> bool:
    try:
        return series.nunique(dropna=False) <= max_unique
    except Exception:
        return False


def is_binary_or_near_binary(series: pd.Series) -> bool:
    try:
        return series.nunique(dropna=False) <= 3
    except Exception:
        return False


def is_discrete_numeric_low_card(series: pd.Series, max_unique: int = 12) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    try:
        return series.nunique(dropna=False) <= max_unique
    except Exception:
        return False


def is_low_information_rhs(col: str, series: pd.Series) -> bool:
    c = col.lower()

    if is_id_like(col) or is_time_like(col):
        return False
    if is_probably_measure_col(col):
        return False
    if is_probably_text_col(series):
        return False

    schema_keywords = [ "level", "code", "type", "status", "flag", "kind", "group", "class", "categorytype",]
    if any(k in c for k in schema_keywords):
        return True
    if is_binary_or_near_binary(series):
        return True
    if is_low_cardinality(series, max_unique=10):
        return True
    if dominant_ratio(series) >= 0.97:
        return True
    if is_discrete_numeric_low_card(series, max_unique=10):
        return True
    return False


def get_fd_drop_tier(col: str, series: pd.Series) -> str:
    c = col.lower()

    if is_id_like(col) or is_time_like(col):
        return "keep"
    if is_probably_measure_col(col):
        return "keep"
    if is_probably_text_col(series):
        return "keep"

    if "level" in c:
        return "hard_drop"
    if c.endswith("hierarchy"):
        return "hard_drop"
    if c.endswith("typecode"):
        return "hard_drop"
    if c.endswith("code"):
        return "soft_drop"
    if "type" in c:
        return "soft_drop"
    if "status" in c:
        return "soft_drop"
    if "flag" in c:
        return "soft_drop"
    if dominant_ratio(series) >= 0.99:
        return "hard_drop"
    if is_binary_or_near_binary(series):
        return "soft_drop"
    if is_low_cardinality(series, max_unique=8):
        return "soft_drop"
    if is_discrete_numeric_low_card(series, max_unique=8):
        return "soft_drop"

    return "keep"