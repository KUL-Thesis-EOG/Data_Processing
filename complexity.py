from functools import partial
import pandas as pd
import numpy as np
import scipy.signal
import antropy
import nolds
import logging
import os
import gc
from tqdm import tqdm
import EntropyHub as EH
import traceback
import time
from typing import Tuple, Optional, Generator, Dict, Any, List
import multiprocessing as mp
import scipy.stats as stats
import pywt
from scipy.fft import fft
from scipy.signal import stft

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("eog_complexity_windowed")

DEFAULT_WINDOW_DURATION_S: float = 2.0
DEFAULT_WINDOW_OVERLAP: float = 0.5
MIN_POINTS_FOR_PSD: int = 10
MIN_POINTS_FACTOR_ENTROPY: int = 2
MAX_WINDOW_LEN_FOR_N2_ALGORITHMS: int = 15000


def _calculate_psd(
    data: np.ndarray, sf: float, nperseg_max: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
    n_points: int = len(data)
    actual_nperseg: int = min(n_points, nperseg_max)

    if n_points < MIN_POINTS_FOR_PSD or actual_nperseg <= 0:
        return None, None, None, 0.0

    try:
        freqs, psd = scipy.signal.welch(data, fs=sf, nperseg=actual_nperseg)
        psd_sum: float = np.sum(psd)

        if psd_sum > 1e-12 and np.isfinite(psd_sum):
            psd_norm: np.ndarray = psd / psd_sum
        else:
            psd_norm = np.zeros_like(psd)
            psd_sum = 0.0

        if not np.all(np.isfinite(psd_norm)):
            psd_norm = np.zeros_like(psd)
            psd_sum = 0.0

        return freqs, psd, psd_norm, psd_sum
    except ValueError:
        return None, None, None, 0.0
    except Exception:
        return None, None, None, 0.0


def window_generator(
    data: np.ndarray, window_size_points: int, window_overlap: float
) -> Generator[Tuple[int, np.ndarray], None, None]:
    n_points: int = len(data)
    if n_points == 0:
        return

    actual_window_size: int = min(n_points, window_size_points)

    if actual_window_size <= 0:
        logger.warning(
            f"Calculated window size ({actual_window_size}) is not positive. No windows generated."
        )
        return
    if not (0.0 <= window_overlap < 1.0):
        logger.warning(
            f"Window overlap must be >= 0 and < 1. Received {window_overlap}. Using 0 overlap."
        )
        window_overlap = 0.0

    hamming_window = np.hamming(actual_window_size)

    step: int = max(1, int(actual_window_size * (1 - window_overlap)))

    for i in range(0, n_points - actual_window_size + 1, step):
        segment = data[i : i + actual_window_size]
        if len(segment) == actual_window_size:
            yield i, segment * hamming_window


def _get_min_len_needed(param_name: str, args: Dict[str, Any]) -> int:
    try:
        if param_name in [
            "Mean",
            "Median",
            "Variance",
            "Skewness",
            "Kurtosis",
            "RMS",
            "ZeroCrossings",
        ]:
            return 4
        elif param_name in ["FFT_Mean", "FFT_Std", "FFT_Max"]:
            return 8
        elif param_name in ["STFT_Mean", "STFT_Std"]:
            nperseg_val = args.get("nperseg", 256)
            return max(16, nperseg_val // 4)
        elif param_name in ["WPT_Mean", "WPT_Std"]:
            maxlevel = args.get("maxlevel", 3)
            return 2 ** (maxlevel + 3)
        elif param_name == "ApEn":
            m: int = args.get("m", 1)
            return (m + 1) * MIN_POINTS_FACTOR_ENTROPY
        elif param_name == "SampEn":
            m: int = args.get("m", 1)
            return (m + 1) * MIN_POINTS_FACTOR_ENTROPY
        elif param_name == "PeEn":
            order: int = args.get("order", 3)
            delay: int = args.get("delay", 1)
            return (order * delay) * MIN_POINTS_FACTOR_ENTROPY
        elif param_name == "FuzEn":
            m: int = args.get("m", 1)
            return (m + 1) * MIN_POINTS_FACTOR_ENTROPY
        elif param_name == "DispEn":
            m: int = args.get("m", 1)
            delay: int = args.get("delay_disp", 1)
            return ((m - 1) * delay + 1) * MIN_POINTS_FACTOR_ENTROPY
        elif param_name in ["TsEn", "ReEn", "ShEn_spectral", "ShEn_manual"]:
            return MIN_POINTS_FOR_PSD
        elif param_name == "LZC":
            return 10
        elif param_name == "DFA":
            order: int = args.get("order", 1)
            return max(20, (order + 1) * 2)
        elif param_name == "HuEx":
            return 30
        elif param_name == "CD":
            emb_dim: int = args.get("emb_dim", 1)
            return emb_dim * 10
        elif param_name == "HFD":
            kmax: int = args.get("kmax", 2)
            return kmax * 3
        else:
            return 1
    except Exception:
        return 1


def calculate_parameter_on_window(
    param_name: str, window_data: np.ndarray, args: Dict[str, Any], sf: float
) -> float:
    n_points: int = len(window_data)
    min_len_needed: int = _get_min_len_needed(param_name, args)

    if n_points < min_len_needed:
        return np.nan

    if np.all(window_data == window_data[0]):
        if param_name in ["ShEn_manual", "ShEn_spectral", "TsEn", "ReEn"]:
            return 0.0
        elif param_name == "LZC":
            return np.nan
        else:
            return np.nan

    std_dev: float = np.std(window_data)
    if std_dev < 1e-10 and param_name in ["ApEn", "SampEn", "FuzEn"]:
        return np.nan

    try:
        result: float = np.nan

        if param_name == "Mean":
            result = np.mean(window_data)
        elif param_name == "Median":
            result = np.median(window_data)
        elif param_name == "Variance":
            result = np.var(window_data)
        elif param_name == "Skewness":
            result = stats.skew(window_data)
        elif param_name == "Kurtosis":
            result = stats.kurtosis(window_data)
        elif param_name == "RMS":
            result = np.sqrt(np.mean(np.square(window_data)))
        elif param_name == "ZeroCrossings":
            result = np.sum(np.diff(np.signbit(window_data).astype(int)))
        elif param_name == "FFT_Mean":
            fft_vals = np.abs(fft(window_data))
            result = np.mean(fft_vals)
        elif param_name == "FFT_Std":
            fft_vals = np.abs(fft(window_data))
            result = np.std(fft_vals)
        elif param_name == "FFT_Max":
            fft_vals = np.abs(fft(window_data))
            result = np.max(fft_vals)
        elif param_name == "STFT_Mean":
            window_type = args.get("window", "hann")
            nperseg_val = args.get("nperseg", 256)
            _, _, Zxx = stft(
                window_data,
                fs=sf,
                window=window_type,
                nperseg=min(n_points, nperseg_val),
            )
            abs_Zxx = np.abs(Zxx)
            result = np.mean(abs_Zxx)
        elif param_name == "STFT_Std":
            window_type = args.get("window", "hann")
            nperseg_val = args.get("nperseg", 256)
            _, _, Zxx = stft(
                window_data,
                fs=sf,
                window=window_type,
                nperseg=min(n_points, nperseg_val),
            )
            abs_Zxx = np.abs(Zxx)
            result = np.std(abs_Zxx)
        elif param_name == "WPT_Mean":
            wavelet = args.get("wavelet", "db4")
            maxlevel = args.get("maxlevel", 3)
            max_allowed_level = int(np.log2(n_points)) - 3
            maxlevel = min(maxlevel, max_allowed_level)
            if maxlevel < 1:
                return np.nan
            try:
                wp = pywt.WaveletPacket(
                    data=window_data,
                    wavelet=wavelet,
                    mode="symmetric",
                    maxlevel=maxlevel,
                )
                coeffs = [node.data for node in wp.get_level(maxlevel, order="natural")]
                if not coeffs:
                    return np.nan
                coeffs = np.concatenate(coeffs)
                result = np.mean(coeffs)
            except Exception:
                result = np.nan
        elif param_name == "WPT_Std":
            wavelet = args.get("wavelet", "db4")
            maxlevel = args.get("maxlevel", 3)
            max_allowed_level = int(np.log2(n_points)) - 3
            maxlevel = min(maxlevel, max_allowed_level)
            if maxlevel < 1:
                return np.nan
            try:
                wp = pywt.WaveletPacket(
                    data=window_data,
                    wavelet=wavelet,
                    mode="symmetric",
                    maxlevel=maxlevel,
                )
                coeffs = [node.data for node in wp.get_level(maxlevel, order="natural")]
                if not coeffs:
                    return np.nan
                coeffs = np.concatenate(coeffs)
                result = np.std(coeffs)
            except Exception:
                result = np.nan
        elif param_name == "ApEn":
            if std_dev > 1e-10:
                result = antropy.app_entropy(
                    window_data, order=args["m"], metric="chebyshev"
                )
        elif param_name == "SampEn":
            if std_dev > 1e-10:
                result = antropy.sample_entropy(
                    window_data, order=args["m"], metric="chebyshev"
                )
        elif param_name == "PeEn":
            result = antropy.perm_entropy(
                window_data, order=args["order"], delay=args["delay"], normalize=True
            )
        elif param_name == "FuzEn":
            r_fuz: Tuple[float, int] = (args["r_factor"] * std_dev, args["n_fuz"])
            if r_fuz[0] > 0:
                eh_result_tuple = EH.FuzzEn(window_data, m=args["m"], r=r_fuz)
                eh_values = eh_result_tuple[1]
                if isinstance(eh_values, (list, np.ndarray)) and len(eh_values) > 0:
                    result = eh_values[-1]
                elif np.isscalar(eh_values):
                    result = eh_values
        elif param_name == "DispEn":
            eh_result_tuple: Tuple[Any, float] = EH.DispEn(
                window_data,
                m=args["m"],
                tau=args["delay_disp"],
                c=args["c_disp"],
                Logx=np.exp(1),
                Typex="NCDF",
                Fluct=False,
                Norm=True,
            )
            result = eh_result_tuple[1]
        elif param_name in ["TsEn", "ReEn", "ShEn_spectral", "ShEn_manual"]:
            nperseg_spec: int = args.get("nperseg", 256)
            _, _, psd_norm, _ = _calculate_psd(window_data, sf, nperseg_spec)
            if psd_norm is None or len(psd_norm) == 0:
                result = np.nan
            else:
                psd_norm = psd_norm[np.isfinite(psd_norm) & (psd_norm >= 0)]
                if len(psd_norm) == 0 or np.sum(psd_norm) < 1e-12:
                    result = 0.0 if param_name != "ReEn" else np.nan
                else:
                    psd_norm = psd_norm / np.sum(psd_norm)
                    if param_name == "TsEn":
                        q: float = args.get("q", 2.0)
                        if abs(q - 1.0) < 1e-9:
                            result = np.nan
                        else:
                            sum_p_q: float = np.sum(psd_norm**q)
                            result = (1.0 / (q - 1.0)) * (1.0 - sum_p_q)
                    elif param_name == "ReEn":
                        beta: float = args.get("beta", 2.0)
                        if abs(beta - 1.0) < 1e-9:
                            result = np.nan
                        else:
                            sum_p_beta: float = np.sum(psd_norm**beta)
                            if sum_p_beta <= 1e-12:
                                result = np.nan
                            else:
                                result = (1.0 / (1.0 - beta)) * np.log(sum_p_beta)
                    elif param_name == "ShEn_manual":
                        psd_nz: np.ndarray = psd_norm[psd_norm > 1e-12]
                        if len(psd_nz) == 0:
                            result = 0.0
                        else:
                            result = -np.sum(psd_nz * np.log(psd_nz))
                    elif param_name == "ShEn_spectral":
                        try:
                            result = antropy.spectral_entropy(
                                window_data,
                                sf=sf,
                                method="welch",
                                nperseg=min(n_points, nperseg_spec),
                                normalize=True,
                                axis=-1,
                            )
                        except ValueError:
                            result = np.nan
                        except Exception:
                            result = np.nan
        elif param_name == "LZC":
            median_val: float = np.median(window_data)
            binary_sequence: np.ndarray = (window_data > median_val).astype(np.uint8)
            if len(binary_sequence) == 0:
                result = np.nan
            else:
                result = antropy.lziv_complexity(binary_sequence, normalize=True)
        elif param_name == "DFA":
            order = args["order"]
            try:
                result = nolds.dfa(window_data, order=order)
            except np.linalg.LinAlgError:
                result = np.nan
            except ValueError:
                result = np.nan
            except Exception:
                result = np.nan
        elif param_name == "HuEx":
            try:
                result = nolds.hurst_rs(window_data)
            except Exception:
                result = np.nan
        elif param_name == "CD":
            emb_dim: int = args["emb_dim"]
            if n_points > MAX_WINDOW_LEN_FOR_N2_ALGORITHMS:
                result = np.nan
            else:
                try:
                    result = nolds.corr_dim(window_data, emb_dim=emb_dim)
                except MemoryError:
                    result = np.nan
                except Exception:
                    result = np.nan
        elif param_name == "HFD":
            kmax: int = args["kmax"]
            try:
                result = antropy.higuchi_fd(window_data, kmax=kmax)
            except Exception:
                result = np.nan

        if not isinstance(result, (int, float, np.number)):
            result = np.nan
        elif not np.isfinite(result):
            pass
        else:
            result = float(result)

        return result

    except MemoryError:
        logger.error(f"MemoryError calculating {param_name} for window size {n_points}")
        return np.nan
    except Exception as e:
        logger.error(
            f"Unhandled error calculating {param_name} for window size {n_points}: {str(e)}\n{traceback.format_exc()}"
        )
        return np.nan


def save_result_to_parquet(
    result_name: str,
    timestamps: pd.DatetimeIndex,
    results_array: np.ndarray,
    output_dir: str,
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    file_path: str = os.path.join(output_dir, f"{result_name}.parquet")
    if len(timestamps) != len(results_array):
        logger.critical(
            f"CRITICAL: Mismatch between timestamp ({len(timestamps)}) and result ({len(results_array)}) counts for {result_name}. Attempting to save with available data."
        )
        min_len = min(len(timestamps), len(results_array))
        timestamps = timestamps[:min_len]
        results_array = results_array[:min_len]
        if len(timestamps) == 0:
            logger.error(
                f"Cannot save {result_name} due to length mismatch resulting in zero common length."
            )
            return None

    df = pd.DataFrame({"timestamp": timestamps, "value": results_array})
    df.set_index("timestamp", inplace=True)
    try:
        df.to_parquet(file_path, engine="pyarrow", compression="snappy", index=True)
    except Exception as e:
        logger.error(
            f"Failed to save {file_path} with pyarrow: {str(e)}. Trying default engine."
        )
        try:
            df.to_parquet(file_path, compression="snappy", index=True)
        except Exception as e2:
            logger.error(f"Fallback save failed for {file_path}: {str(e2)}")
            file_path = None
    del df
    gc.collect()
    return file_path


def save_raw_windows_to_parquet(
    channel: str,
    timestamps: pd.DatetimeIndex,
    window_data_list: List[np.ndarray],
    output_dir: str,
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    file_path: str = os.path.join(output_dir, f"raw_windows_{channel}.parquet")
    num_windows = len(window_data_list)
    if len(timestamps) != num_windows:
        logger.error(
            f"Mismatch between number of timestamps ({len(timestamps)}) and number of windows ({num_windows}) for raw data saving of channel {channel}. Skipping raw save."
        )
        return None

    df_data = {
        "window_index": range(num_windows),
        "timestamp": timestamps,
        "values": window_data_list,
    }
    df = pd.DataFrame(df_data)
    df.set_index("window_index", inplace=True)

    try:
        df.to_parquet(file_path, engine="pyarrow", compression="snappy", index=True)
        logger.info(f"Saved raw windowed data for {channel} to {file_path}")
    except Exception as e:
        logger.error(
            f"Failed to save raw window data for {channel} to {file_path}: {e}. Trying default engine."
        )
        try:
            df.to_parquet(file_path, compression="snappy", index=True)
            logger.info(
                f"Saved raw windowed data for {channel} to {file_path} (using default engine)."
            )
        except Exception as e2:
            logger.error(f"Fallback save failed for raw windows file {file_path}: {e2}")
            file_path = None

    del df, df_data
    gc.collect()
    return file_path


def process_window_for_parameter(window_with_idx_and_start, param_name, args, sf):
    window_idx, (start_idx, window_data) = window_with_idx_and_start
    result = calculate_parameter_on_window(param_name, window_data, args, sf)
    return window_idx, start_idx, result


def process_channel(
    channel: str,
    data: np.ndarray,
    index: pd.DatetimeIndex,
    param_configs: Dict[str, Dict[str, Any]],
    sf: float,
    output_dir: str,
    window_duration_s: float,
    window_overlap: float,
) -> Tuple[List[str], List[str]]:
    param_saved_files: List[str] = []
    raw_saved_files: List[str] = []
    n_points_total: int = len(data)
    window_size_points: int = int(sf * window_duration_s)

    if n_points_total == 0:
        logger.warning(f"No data points for channel {channel}. Skipping.")
        return [], []
    if window_size_points <= 0:
        logger.error(
            f"Window size must be positive ({window_size_points} points). Skipping channel {channel}."
        )
        return [], []
    if len(index) != n_points_total:
        logger.error(
            f"Data length ({n_points_total}) and index length ({len(index)}) mismatch for channel {channel}. Skipping."
        )
        return [], []

    logger.info(
        f"Windowing channel {channel} (length: {n_points_total}) into {window_duration_s}s ({window_size_points}pts) windows with {window_overlap*100:.0f}% overlap."
    )

    start_time_windowing: float = time.time()
    windows: List[Tuple[int, np.ndarray]] = list(
        window_generator(data, window_size_points, window_overlap)
    )
    end_time_windowing: float = time.time()
    num_windows = len(windows)
    logger.info(
        f"Generated {num_windows} windows for {channel} in {end_time_windowing - start_time_windowing:.2f} seconds."
    )

    if not windows:
        logger.warning(
            f"Windowing resulted in no windows for channel {channel}. Skipping calculation."
        )
        return [], []

    # Store window data and start indices separately for raw saving and timestamp lookup
    all_start_indices = np.array([w[0] for w in windows], dtype=np.int64)
    all_window_data = [w[1] for w in windows]  # Keep as list of arrays

    # Generate timestamps for parameter results and raw windows
    valid_start_indices_mask = all_start_indices < len(index)
    if not np.all(valid_start_indices_mask):
        logger.error(
            f"Found start indices out of bounds for the original time index. Clamping."
        )
        all_start_indices = np.clip(all_start_indices, 0, len(index) - 1)

    if len(all_start_indices) > 0:
        timestamps_array = index[all_start_indices]
    else:
        timestamps_array = pd.DatetimeIndex([])  # Should not happen if windows exist

    # Save raw windowed data first
    raw_file_path = save_raw_windows_to_parquet(
        channel, timestamps_array, all_window_data, output_dir
    )
    if raw_file_path:
        raw_saved_files.append(raw_file_path)

    # Proceed with parameter calculation
    num_cores = mp.cpu_count()
    pool_size = max(1, min(num_cores - 1 if num_cores > 1 else 1, num_windows))
    logger.info(f"Using {pool_size} CPU cores for parallel parameter processing")

    windows_with_indices_and_start = list(
        enumerate(windows)
    )  # Format: (window_idx, (start_idx, window_data))

    with mp.Pool(processes=pool_size) as pool:
        for param_name, args in tqdm(
            param_configs.items(),
            desc=f"Calculating parameters for {channel}",
            leave=False,
        ):
            window_results_array = np.full(num_windows, np.nan, dtype=np.float64)

            process_func = partial(
                process_window_for_parameter, param_name=param_name, args=args, sf=sf
            )
            chunk_size = max(
                1, min(100, num_windows // (pool_size * 4 if pool_size > 0 else 1))
            )
            if chunk_size == 0:
                chunk_size = 1

            try:
                map_results = pool.map(
                    process_func, windows_with_indices_and_start, chunksize=chunk_size
                )

                processed_indices_count = 0
                for (
                    win_idx,
                    _,
                    result_val,
                ) in map_results:
                    if win_idx >= 0 and win_idx < num_windows:
                        window_results_array[win_idx] = result_val
                        processed_indices_count += 1
                    else:
                        logger.warning(
                            f"Received invalid window index {win_idx} for parameter {param_name}. Max index is {num_windows-1}."
                        )

                if processed_indices_count != num_windows:
                    logger.warning(
                        f"Parameter {param_name}: Expected {num_windows} results, but processed {processed_indices_count}."
                    )

                if len(timestamps_array) == len(window_results_array):
                    result_name: str = f"{param_name}_{channel}"
                    file_path: Optional[str] = save_result_to_parquet(
                        result_name, timestamps_array, window_results_array, output_dir
                    )
                    if file_path:
                        param_saved_files.append(file_path)
                else:
                    logger.error(
                        f"Timestamp array length ({len(timestamps_array)}) mismatch with results ({len(window_results_array)}) for {param_name}. Skipping save."
                    )

            except Exception as e:
                logger.error(
                    f"Error processing parameter {param_name} for channel {channel}: {e}\n{traceback.format_exc()}"
                )

            del window_results_array
            gc.collect()

    del (
        windows,
        windows_with_indices_and_start,
        all_start_indices,
        all_window_data,
        timestamps_array,
    )
    gc.collect()

    return param_saved_files, raw_saved_files


def load_channel_data(
    input_file: str, channel: str
) -> Tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
    try:
        temp_df: pd.DataFrame = pd.read_parquet(input_file)
        if channel not in temp_df.columns:
            logger.error(f"Channel '{channel}' not found in {input_file}")
            return None, None
        if not isinstance(temp_df.index, pd.DatetimeIndex):
            logger.error(f"Input file {input_file} does not have a DatetimeIndex.")
            try:
                temp_df.index = pd.to_datetime(temp_df.index)
                logger.info("Converted index to DatetimeIndex.")
            except Exception as idx_e:
                logger.error(f"Failed to convert index to DatetimeIndex: {idx_e}")
                return None, None

        temp_df.dropna(subset=[channel], inplace=True)

        if temp_df.empty:
            logger.warning(
                f"No valid data for channel '{channel}' after dropping NaNs."
            )
            return np.array([], dtype=np.float64), pd.DatetimeIndex([])

        data_index: pd.DatetimeIndex = temp_df.index
        data_values: np.ndarray = pd.to_numeric(
            temp_df[channel], errors="coerce"
        ).to_numpy(dtype=np.float64)

        nan_mask = np.isnan(data_values)
        if np.any(nan_mask):
            logger.warning(
                f"Found {np.sum(nan_mask)} NaNs in channel '{channel}' after initial dropna and numeric conversion. Removing corresponding entries."
            )
            data_values = data_values[~nan_mask]
            data_index = data_index[~nan_mask]

        del temp_df
        gc.collect()
        if len(data_values) != len(data_index):
            logger.error(
                "CRITICAL: Length mismatch between data and index after processing. Returning None."
            )
            return None, None

        return data_values, data_index

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        return None, None
    except Exception as e:
        logger.error(
            f"Error loading data for channel '{channel}' from {input_file}: {str(e)}"
        )
        return None, None


def calculate_eog_complexity_windowed(
    input_file: str,
    output_dir: str,
    sf: int = 200,
    window_duration_s: float = DEFAULT_WINDOW_DURATION_S,
    window_overlap: float = DEFAULT_WINDOW_OVERLAP,
) -> Tuple[List[str], List[str], int, int]:

    required_cols: List[str] = ["horizontal_filtered", "vertical_filtered"]
    all_param_saved_files: List[str] = []
    all_raw_saved_files: List[str] = []
    param_configs: Dict[str, Dict[str, Any]] = {
        "Mean": {},
        "Median": {},
        "Variance": {},
        "Skewness": {},
        "Kurtosis": {},
        "RMS": {},
        "ZeroCrossings": {},
        "FFT_Mean": {},
        "FFT_Std": {},
        "FFT_Max": {},
        "STFT_Mean": {"window": "hann", "nperseg": 256},
        "STFT_Std": {"window": "hann", "nperseg": 256},
        "WPT_Mean": {"wavelet": "db4", "maxlevel": 3},
        "WPT_Std": {"wavelet": "db4", "maxlevel": 3},
        "ApEn": {"m": 2, "r_factor": 0.2},
        "SampEn": {"m": 2, "r_factor": 0.2},
        "PeEn": {"order": 3, "delay": 1},
        "DispEn": {"m": 2, "c_disp": 6, "delay_disp": 1},
        "TsEn": {"q": 2.0, "nperseg": 256},
        "ReEn": {"beta": 2.0, "nperseg": 256},
        "ShEn_spectral": {"nperseg": 256},
        "ShEn_manual": {"nperseg": 256},
        "LZC": {},
        "DFA": {"order": 1},
        "HuEx": {},
        "CD": {"emb_dim": 2},
        "HFD": {"kmax": 10},
        "FuzEn": {"m": 2, "r_factor": 0.2, "n_fuz": 2},
    }
    os.makedirs(output_dir, exist_ok=True)

    files_checked_count: int = 0
    total_params_expected_per_channel: int = len(param_configs)
    total_param_files_saved: int = 0
    total_raw_files_saved: int = 0

    for channel in required_cols:
        logger.info(f"--- Processing Channel: {channel} ---")
        logger.info(f"Loading data for {channel}...")
        data, index = load_channel_data(input_file, channel)

        if (
            data is not None
            and index is not None
            and len(data) > 0
            and len(index) == len(data)
        ):
            logger.info(
                f"Calculating parameters and saving raw windows for {channel} (length: {len(data)} points)..."
            )
            channel_param_files, channel_raw_files = process_channel(
                channel,
                data,
                index,
                param_configs,
                sf,
                output_dir,
                window_duration_s,
                window_overlap,
            )
            all_param_saved_files.extend(channel_param_files)
            all_raw_saved_files.extend(channel_raw_files)
            total_param_files_saved += len(channel_param_files)
            total_raw_files_saved += len(channel_raw_files)

            logger.info(f"Quick check of saved parameter files for {channel}...")
            for file_path in tqdm(
                channel_param_files, desc=f"Checking {channel} param files", leave=False
            ):
                if os.path.exists(file_path):
                    try:
                        df_check: pd.DataFrame = pd.read_parquet(file_path)
                        if (
                            not df_check.empty
                            and df_check.index.name == "timestamp"
                            and "value" in df_check.columns
                        ):
                            files_checked_count += 1
                        else:
                            logger.warning(
                                f"Parameter file {file_path} seems invalid, empty, or lacks timestamp index/value column."
                            )
                        del df_check
                    except Exception as e:
                        logger.error(
                            f"Error reading/checking parameter result file {file_path}: {str(e)}"
                        )
                else:
                    logger.warning(
                        f"Parameter file {file_path} reported as saved but not found."
                    )

            # Optional: Add check for raw files if needed
            # for file_path in channel_raw_files: ...

            del data, index, channel_param_files, channel_raw_files
            gc.collect()
        elif data is None or index is None:
            logger.error(
                f"Failed to load data or index for channel {channel}. Skipping."
            )
        else:
            logger.warning(
                f"Skipping channel {channel} due to no valid data points after loading/cleaning."
            )

        logger.info(f"--- Completed Channel: {channel} ---")

    total_expected_params: int = len(required_cols) * total_params_expected_per_channel
    logger.info(
        f"Finished processing. Total expected parameter files: {total_expected_params}. "
        f"Parameter files saved: {total_param_files_saved}. Files passing basic check: {files_checked_count}. "
        f"Raw window files saved: {total_raw_files_saved}."
    )

    return (
        all_param_saved_files,
        all_raw_saved_files,
        files_checked_count,
        total_expected_params,
    )


def create_summary_file(
    param_saved_files: List[str],  # Only parameter files
    raw_saved_files: List[str],  # Raw files listed separately
    valid_param_file_count: int,
    total_expected_params: int,
    output_dir: str,
) -> Tuple[str, str]:
    summary_data: Dict[str, Any] = {
        "total_parameters_expected": total_expected_params,
        "parameter_files_saved_count": len(param_saved_files),
        "parameter_files_passing_basic_check": valid_param_file_count,
        "raw_window_files_saved_count": len(raw_saved_files),
        "saved_parameter_files_list": [os.path.basename(f) for f in param_saved_files],
        "saved_raw_window_files_list": [os.path.basename(f) for f in raw_saved_files],
    }

    summary_df = pd.DataFrame([summary_data])
    summary_parquet_path: str = os.path.join(output_dir, "processing_summary.parquet")
    summary_txt_path: str = os.path.join(output_dir, "processing_summary.txt")

    try:
        summary_df.to_parquet(summary_parquet_path, index=False)
        logger.info(f"Summary saved to {summary_parquet_path}")
    except Exception as e:
        logger.error(f"Failed to save summary parquet: {str(e)}")

    try:
        with open(summary_txt_path, "w") as f:
            f.write(f"Total parameters expected: {total_expected_params}\n")
            f.write(
                f"Parameter parquet files saved (reported): {len(param_saved_files)}\n"
            )
            f.write(f"Parameter files passing basic check: {valid_param_file_count}\n")
            f.write(
                f"Raw window parquet files saved (reported): {len(raw_saved_files)}\n"
            )
            f.write("\nList of saved parameter files reported:\n")
            for fpath in param_saved_files:
                f.write(f"- {os.path.basename(fpath)}\n")
            f.write("\nList of saved raw window files reported:\n")
            for fpath in raw_saved_files:
                f.write(f"- {os.path.basename(fpath)}\n")
        logger.info(f"Text summary saved to {summary_txt_path}")
    except Exception as e:
        logger.error(f"Failed to save text summary: {str(e)}")

    return summary_parquet_path, summary_txt_path


if __name__ == "__main__":
    participant: str = "sofia"
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    data_dir: str = os.path.join(base_dir, "data")
    results_base_dir: str = os.path.join(base_dir, "results")

    input_file: str = os.path.join(data_dir, f"{participant}_processed_data.parquet")
    output_dir: str = os.path.join(
        results_base_dir, f"{participant}_eog_complexity_windowed"
    )

    sampling_frequency: int = 200
    window_duration: float = 2.0
    window_overlap: float = 0.5  # Reintroduced overlap

    start_run_time: float = time.time()
    logger.info(f"Starting windowed EOG complexity calculation for: {participant}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Sampling Frequency: {sampling_frequency} Hz")
    logger.info(f"Window Duration: {window_duration} s")
    logger.info(f"Window Overlap: {window_overlap*100:.0f}%")

    os.makedirs(output_dir, exist_ok=True)

    all_param_files: List[str]
    all_raw_files: List[str]
    checked_param_file_count: int
    total_expected_params: int

    all_param_files, all_raw_files, checked_param_file_count, total_expected_params = (
        calculate_eog_complexity_windowed(
            input_file,
            output_dir,
            sf=sampling_frequency,
            window_duration_s=window_duration,
            window_overlap=window_overlap,
        )
    )

    logger.info("Creating summary file...")
    create_summary_file(
        all_param_files,
        all_raw_files,
        checked_param_file_count,
        total_expected_params,
        output_dir,
    )

    end_run_time: float = time.time()
    logger.info("Processing complete.")
    logger.info(f"Total time: {end_run_time - start_run_time:.2f} seconds")
    logger.info(
        f"Expected parameter files: {total_expected_params}. Parameter files saved (reported): {len(all_param_files)}. "
        f"Parameter files passing check: {checked_param_file_count}. Raw window files saved: {len(all_raw_files)}."
    )
    logger.info(f"Individual parameter results saved in {output_dir}")
    logger.info(f"Raw windowed signal data saved in {output_dir}")
    logger.info(f"Summary files saved in {output_dir}")
