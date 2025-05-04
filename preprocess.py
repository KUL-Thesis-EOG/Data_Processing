import pandas as pd
import numpy as np


def get_experiment_mask(df_res):
    """
    Extract the experiment mask (breaks) from df_res.

    This function identifies experiment breaks by looking for columns matching
    'break\d+.duration' pattern, concatenates their values, and creates a
    time-series mask indicating when breaks occurred.

    Args:
        df_res: DataFrame with experiment results containing break information

    Returns:
        DataFrame with DatetimeIndex and 'is_break' boolean column indicating
        whether each second is part of a break period
    """
    cols_to_join = df_res.filter(regex=r"^break\d+\.duration$").columns
    df_res["breaks"] = df_res[cols_to_join].fillna("").astype(str).agg("".join, axis=1)

    df_breaks_filtered = df_res.copy()
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
    """
    Extract ISA (Instantaneous Self-Assessment) responses from df_res.

    This function extracts slider responses, resamples them to 1-second intervals,
    and interpolates missing values using quadratic interpolation.

    Args:
        df_res: DataFrame with experiment results containing ISA responses

    Returns:
        Series with timestamp index and ISA response values
    """
    return (
        df_res["slider.response"]
        .rename("F_ISA")
        .resample("1s")
        .nearest(limit=1)
        .interpolate(method="quadratic")
    )


def preprocess_features(participant):
    """
    Load and preprocess feature data for a participant.

    This function loads EOG (electrooculography) complexity windowed features
    from parquet files and combines them into a single DataFrame.

    Args:
        participant: String identifier for the participant

    Returns:
        DataFrame with features resampled to 1-second intervals
    """
    df_feat = pd.DataFrame()

    for x in pd.read_parquet(
        f"results/{participant}_eog_complexity_windowed/processing_summary.parquet"
    )["saved_parameter_files_list"][0]:
        df_feat[x.removesuffix(".parquet")] = pd.read_parquet(
            f"results/{participant}_eog_complexity_windowed/{x}"
        )

    return df_feat.resample("1s").first()


def apply_breaks_and_grouping(df_breaks, df_data, aggregate):
    """
    Apply break masking and group data into aggregate segments.

    This function removes data during break periods and groups the remaining
    data into segments of a specified size, keeping only complete groups.

    Args:
        df_breaks: DataFrame with 'is_break' boolean column
        df_data: DataFrame with the data to be masked and grouped
        aggregate: Integer size of each group (in seconds)

    Returns:
        tuple: (df_full, grouping_key_full) - filtered data and corresponding group indices
    """
    df_masked = df_data.mask(df_breaks.is_break).dropna()

    grouping_key = np.arange(len(df_masked)) // aggregate

    group_counts = df_masked.groupby(grouping_key).size()

    full_groups = group_counts[group_counts == aggregate].index

    mask_full_groups = pd.Series(grouping_key).isin(full_groups).values

    df_full = df_masked[mask_full_groups]
    grouping_key_full = grouping_key[mask_full_groups]

    return df_full, grouping_key_full


def preprocess_features_flat(df_breaks, participant: str, aggregate: int):
    """
    Preprocess features into a flat format with aggregated statistics.

    This function loads features, applies break masking, groups data into
    segments, and computes various statistical aggregations for each segment.

    Args:
        df_breaks: DataFrame with 'is_break' boolean column
        participant: String identifier for the participant
        aggregate: Integer size of each group (in seconds)

    Returns:
        DataFrame with aggregated features for each segment
    """
    df_feat = preprocess_features(participant)

    df_feat_full, grouping_key_full = apply_breaks_and_grouping(
        df_breaks, df_feat, aggregate
    )

    aggregations = {
        "mean": "mean",
        "median": "median",
        "min": "min",
        "max": "max",
        "variance": "var",
        "q1": lambda x: x.quantile(0.25),
        "q3": lambda x: x.quantile(0.75),
    }

    unique_groups = sorted(set(grouping_key_full))
    x_full = pd.DataFrame(index=unique_groups)

    for k, v in aggregations.items():
        agg_result = df_feat_full.groupby(grouping_key_full).agg(v)
        agg_result.columns = [f"{col}_{k}" for col in agg_result.columns]
        x_full = x_full.join(agg_result)

    return x_full


def preprocess_isa_flat(df_breaks, df_isa, aggregate: int, binary=True):
    """
    Preprocess ISA responses into a flat format.

    This function applies break masking, groups data, and optionally converts
    to binary classification based on a threshold.

    Args:
        df_breaks: DataFrame with 'is_break' boolean column
        df_isa: Series with ISA responses
        aggregate: Integer size of each group (in seconds)
        binary: Boolean indicating whether to convert to binary (default: True)

    Returns:
        Series with ISA responses (averaged per segment)
    """
    df_isa_full, grouping_key_full = apply_breaks_and_grouping(
        df_breaks, df_isa, aggregate
    )

    y_full = df_isa_full.groupby(grouping_key_full).mean()

    if binary:
        y_full = y_full.map(lambda x: x > 3)  # Threshold at 3 for fatigued

    return y_full


def preprocess_data_flat(participant: str, aggregate: int, binary=True):
    """
    Main preprocessing function that combines features and ISA responses.

    This function loads raw data, preprocesses features and ISA responses,
    and returns them in a format suitable for machine learning.

    Args:
        participant: String identifier for the participant
        aggregate: Integer size of each group (in seconds)
        binary: Boolean indicating whether to convert ISA to binary (default: True)

    Returns:
        tuple: (x_full, y_full) - features and target values
    """
    df_res = pd.read_parquet(f"data/{participant}_pavlovia_raw_data.parquet").set_index(
        "timestamp"
    )

    # Extract breaks and ISA responses from df_res
    df_breaks = get_experiment_mask(df_res)
    df_isa = get_isa_responses(df_res)

    # Process features and ISA separately, avoid leakage
    x_full = preprocess_features_flat(df_breaks, participant, aggregate)
    y_full = preprocess_isa_flat(df_breaks, df_isa, aggregate, binary)

    return x_full, y_full
