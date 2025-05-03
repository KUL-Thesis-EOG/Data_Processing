import pandas as pd
import numpy as np


def get_experiment_mask(df_res):
    """Extract the experiment mask (breaks) from df_res."""
    cols_to_join = df_res.filter(regex=r"^break\d+\.duration$").columns
    df_res["breaks"] = df_res[cols_to_join].fillna("").astype(str).agg("".join, axis=1)

    df_breaks_filtered = df_res.dropna(subset=["breaks"]).copy()
    df_breaks_filtered["duration_seconds"] = pd.to_numeric(
        df_breaks_filtered["breaks"], errors="coerce"
    )
    df_breaks_filtered = df_breaks_filtered[df_breaks_filtered["duration_seconds"] > 0]

    df_breaks = pd.DataFrame(
        {"is_break": []}, index=pd.DatetimeIndex([], freq="s")
    ).astype(bool)

    if not df_breaks_filtered.empty:
        df_breaks_filtered["duration_ceil_s"] = np.ceil(
            df_breaks_filtered["duration_seconds"]
        )
        df_breaks_filtered["t_start"] = df_breaks_filtered.index - pd.to_timedelta(
            df_breaks_filtered["duration_ceil_s"] - 1, unit="s"
        )

        min_time = df_breaks_filtered["t_start"].min()
        max_time = df_breaks_filtered.index.max()

        if not pd.isna(min_time) and not pd.isna(max_time):
            full_index = pd.date_range(start=min_time, end=max_time, freq="s")
            df_breaks = pd.DataFrame({"is_break": False}, index=full_index)

            for break_end_time, row in df_breaks_filtered.iterrows():
                duration_ceil = row["duration_ceil_s"]
                break_start_time = break_end_time - pd.to_timedelta(
                    duration_ceil - 1, unit="s"
                )
                df_breaks.loc[break_start_time:break_end_time, "is_break"] = True

    return df_breaks.resample("1s").first()


def get_isa_responses(df_res):
    """Extract ISA responses from df_res."""
    return (
        df_res["slider.response"]
        .rename("F_ISA")
        .resample("1s")
        .nearest(limit=1)
        .interpolate(method="quadratic")
    )


def preprocess_features(participant):
    """Load and preprocess feature data."""
    df_feat = pd.DataFrame()

    for x in pd.read_parquet(
        f"results/{participant}_eog_complexity_windowed/processing_summary.parquet"
    )["saved_parameter_files_list"][0]:
        df_feat[x.removesuffix(".parquet")] = pd.read_parquet(
            f"results/{participant}_eog_complexity_windowed/{x}"
        )

    return df_feat.resample("1s").first()


def preprocess_data_flat(
    participant: str,
    aggregate: int,
):
    # Load raw experiment data
    df_res = pd.read_parquet(f"data/{participant}_pavlovia_raw_data.parquet").set_index(
        "timestamp"
    )

    df_breaks = get_experiment_mask(df_res)

    df_isa = get_isa_responses(df_res)

    df_feat = preprocess_features(participant)

    df_final = df_feat.join(df_isa).mask(df_breaks.is_break).dropna()

    grouping_key = np.arange(len(df_final)) // aggregate

    group_counts = df_final.groupby(grouping_key).size()

    full_groups = group_counts[group_counts == aggregate].index

    mask_full_groups = pd.Series(grouping_key).isin(full_groups).values

    df_final_full = df_final[mask_full_groups]
    grouping_key_full = grouping_key[mask_full_groups]

    aggregations = {
        "mean": "mean",
        "median": "median",
        "min": "min",
        "max": "max",
        "variance": "var",
        "q1": lambda x: x.quantile(0.25),
        "q3": lambda x: x.quantile(0.75),
    }

    df_x = df_final_full.drop("F_ISA", axis=1)

    x_full = pd.DataFrame(df_x.groupby(grouping_key_full).agg("mean"))
    y_full = (
        df_final_full["F_ISA"].groupby(grouping_key_full).mean().map(lambda x: x > 3)
    )

    for k, v in aggregations.items():
        x_full = x_full.join(df_x.groupby(grouping_key_full).agg(v), rsuffix=k)

    return x_full, y_full
