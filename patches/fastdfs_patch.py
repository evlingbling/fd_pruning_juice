def patch_fastdfs_canonicalize_types(fastdfs_type_transform):
    if not hasattr(fastdfs_type_transform, "CanonicalizeTypes"):
        print("CanonicalizeTypes not found")
        return

    cls = fastdfs_type_transform.CanonicalizeTypes
    original_call = cls.__call__

    if getattr(original_call, "_fd_patch_applied", False):
        return

    def _filter_meta_columns(meta_columns, actual_cols_set):
        if meta_columns is None:
            return meta_columns

        new_cols = []
        for col in meta_columns:
            if hasattr(col, "name"):
                if col.name in actual_cols_set:
                    new_cols.append(col)
            elif isinstance(col, str):
                if col in actual_cols_set:
                    new_cols.append(col)
            else:
                new_cols.append(col)
        return new_cols

    def _filter_value(v, actual_cols_set):
        if isinstance(v, dict):
            return {
                k: vv for k, vv in v.items()
                if (not isinstance(k, str)) or (k in actual_cols_set)
            }
        if isinstance(v, list):
            out = []
            for x in v:
                if hasattr(x, "name"):
                    if x.name in actual_cols_set:
                        out.append(x)
                elif isinstance(x, str):
                    if x in actual_cols_set:
                        out.append(x)
                else:
                    out.append(x)
            return out
        if isinstance(v, tuple):
            out = []
            for x in v:
                if hasattr(x, "name"):
                    if x.name in actual_cols_set:
                        out.append(x)
                elif isinstance(x, str):
                    if x in actual_cols_set:
                        out.append(x)
                else:
                    out.append(x)
            return tuple(out)
        if isinstance(v, set):
            out = set()
            for x in v:
                if hasattr(x, "name"):
                    if x.name in actual_cols_set:
                        out.add(x)
                elif isinstance(x, str):
                    if x in actual_cols_set:
                        out.add(x)
                else:
                    try:
                        out.add(x)
                    except Exception:
                        pass
            return out
        return v

    def wrapped(self, table_df, table_metadata, *args, **kwargs):
        actual_cols_set = set(table_df.columns)

        if hasattr(table_metadata, "columns"):
            try:
                table_metadata.columns = _filter_meta_columns(
                    table_metadata.columns,
                    actual_cols_set,
                )
            except Exception:
                pass

        for attr in [
            "column_names", "column_types", "logical_types", "semantic_tags",
            "dtypes", "dtype_map", "logical_type_map",
            "data_types", "feature_types", "col_types",
            "schema", "column_metadata", "column_stats", "ww_schema"
        ]:
            if hasattr(table_metadata, attr):
                try:
                    setattr(
                        table_metadata,
                        attr,
                        _filter_value(getattr(table_metadata, attr), actual_cols_set),
                    )
                except Exception:
                    pass

        if hasattr(table_metadata, "__dict__"):
            for k, v in list(table_metadata.__dict__.items()):
                try:
                    if k == "columns":
                        table_metadata.__dict__[k] = _filter_meta_columns(v, actual_cols_set)
                    else:
                        table_metadata.__dict__[k] = _filter_value(v, actual_cols_set)
                except Exception:
                    pass

        return original_call(self, table_df, table_metadata, *args, **kwargs)

    wrapped._fd_patch_applied = True
    cls.__call__ = wrapped
