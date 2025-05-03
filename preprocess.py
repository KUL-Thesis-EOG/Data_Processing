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
    return df_res["slider.response"].rename("F_ISA").bfill().resample("1s").ffill()


def preprocess_raw_eog(participant):
    """Load and preprocess raw EOG data."""
    # Load raw EOG data from processed data file
    df_eog = pd.read_parquet(f"data/{participant}_processed_data.parquet")

    # Select the EOG channels
    eog_channels = ["horizontal_filtered", "vertical_filtered"]
    df_eog = df_eog[eog_channels]

    # Resample to 1s if needed (the processed data might already be at a different sampling rate)
    return df_eog.resample("1s").first()


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

    df_eog = preprocess_raw_eog(participant)

    df_final = df_eog.join(df_isa).mask(df_breaks.is_break).dropna()

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


def preprocess_data(
    participant: str,
    aggregate: int,
):
    """
    Preprocess raw EOG data for deep learning models without flattening.

    Args:
        participant: Participant ID
        aggregate: Window size (number of samples per window)

    Returns:
        X: numpy array of shape (n_samples, window_size, n_features)
        y: numpy array of shape (n_samples,) with binary labels (ISA > 3)
    """
    # Load raw experiment data
    df_res = pd.read_parquet(f"data/{participant}_pavlovia_raw_data.parquet").set_index(
        "timestamp"
    )

    df_breaks = get_experiment_mask(df_res)

    df_isa = get_isa_responses(df_res)

    # Load raw EOG signals instead of features
    df_eog = preprocess_raw_eog(participant)

    # Join with ISA and mask by breaks
    df_final = df_eog.join(df_isa).mask(df_breaks.is_break).dropna()

    grouping_key = np.arange(len(df_final)) // aggregate

    group_counts = df_final.groupby(grouping_key).size()

    full_groups = group_counts[group_counts == aggregate].index

    mask_full_groups = pd.Series(grouping_key).isin(full_groups).values

    df_final_full = df_final[mask_full_groups]
    grouping_key_full = grouping_key[mask_full_groups]

    # Prepare features (excluding the target column)
    df_x = df_final_full.drop("F_ISA", axis=1)

    # Get the number of samples, features, and unique groups
    n_samples = len(full_groups)
    n_features = df_x.shape[1]  # This will be 2 for horizontal and vertical EOG

    # Create empty array for windowed data
    x_full = np.zeros((n_samples, aggregate, n_features))

    # Fill the array with windowed data
    for i, group_id in enumerate(full_groups):
        mask = grouping_key_full == group_id
        x_full[i] = df_x[mask].values

    # Create target array (binary classification based on mean ISA > 3)
    y_full = (
        df_final_full["F_ISA"]
        .groupby(grouping_key_full)
        .mean()
        .map(lambda x: x > 3)
        .values
    )

    return x_full, y_full
