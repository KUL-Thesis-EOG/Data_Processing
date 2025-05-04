import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import os


def plot_signals_interactive(
    all_data,
    vertical_raw_col="vertical_value",
    vertical_filtered_col="vertical_filtered",
    vertical_smoothed_col=None,
    horizontal_raw_col="horizontal_value",
    horizontal_filtered_col="horizontal_filtered",
    horizontal_smoothed_col=None,
    n_samples=5000,
    title="Signal Comparison",
):
    """Plot signals with seaborn styling."""
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    n_samples = min(n_samples, len(all_data))
    data_subset = all_data.iloc[:n_samples]
    x_axis = (
        data_subset.index
        if isinstance(data_subset.index, pd.DatetimeIndex)
        else np.arange(n_samples)
    )
    x_title = "Time" if isinstance(data_subset.index, pd.DatetimeIndex) else "Samples"

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=16, y=0.98)

    # Plot vertical signals
    ax1.set_title("Vertical Signal Comparison", fontsize=14, pad=10)
    if vertical_raw_col in data_subset.columns:
        ax1.plot(x_axis, data_subset[vertical_raw_col], label="Raw Vertical")

    if vertical_filtered_col in data_subset.columns:
        ax1.plot(x_axis, data_subset[vertical_filtered_col], label="Filtered Vertical")

    if (
        vertical_smoothed_col is not None
        and vertical_smoothed_col in data_subset.columns
    ):
        ax1.plot(x_axis, data_subset[vertical_smoothed_col], label="Smoothed Vertical")

    ax1.set_ylabel("Amplitude", fontsize=12)
    ax1.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax1.grid(True)

    # Set x-axis limits to match the data range exactly
    ax1.set_xlim(x_axis[0], x_axis[-1])

    # Plot horizontal signals
    ax2.set_title("Horizontal Signal Comparison", fontsize=14, pad=10)
    if horizontal_raw_col in data_subset.columns:
        ax2.plot(x_axis, data_subset[horizontal_raw_col], label="Raw Horizontal")

    if horizontal_filtered_col in data_subset.columns:
        ax2.plot(
            x_axis, data_subset[horizontal_filtered_col], label="Filtered Horizontal"
        )

    if (
        horizontal_smoothed_col is not None
        and horizontal_smoothed_col in data_subset.columns
    ):
        ax2.plot(
            x_axis, data_subset[horizontal_smoothed_col], label="Smoothed Horizontal"
        )

    ax2.set_xlabel(x_title, fontsize=12)
    ax2.set_ylabel("Amplitude", fontsize=12)
    ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax2.grid(True)

    # Set x-axis limits to match the data range exactly
    ax2.set_xlim(x_axis[0], x_axis[-1])

    plt.tight_layout()
    plt.close()  # Close the figure to prevent double display
    return fig


def plot_fft_vertical_horizontal(
    vertical_value, horizontal_value, fs=1000, freq_limit=100, disp_peaks=False
):
    """Plot FFT with seaborn styling."""
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    if isinstance(vertical_value, pd.Series):
        vertical_value = vertical_value.values
    if isinstance(horizontal_value, pd.Series):
        horizontal_value = horizontal_value.values

    vertical_value_centered = vertical_value - np.mean(vertical_value)
    horizontal_value_centered = horizontal_value - np.mean(horizontal_value)

    N = len(vertical_value_centered)
    if N == 0:
        print("Warning: Input data has zero length.")
        return None

    if len(horizontal_value_centered) != N:
        raise ValueError("Vertical and horizontal signals must have the same length.")

    vertical_fft = np.fft.fft(vertical_value_centered)
    horizontal_fft = np.fft.fft(horizontal_value_centered)

    frequencies = np.fft.fftfreq(N, 1 / fs)

    positive_freq_mask = frequencies >= 0
    positive_frequencies = frequencies[positive_freq_mask]
    vertical_fft_positive = vertical_fft[positive_freq_mask]
    horizontal_fft_positive = horizontal_fft[positive_freq_mask]

    actual_freq_limit = min(freq_limit, positive_frequencies.max())
    limited_indices = positive_frequencies <= actual_freq_limit

    positive_frequencies_limited = positive_frequencies[limited_indices]
    vertical_magnitude_limited = np.abs(vertical_fft_positive[limited_indices])
    horizontal_magnitude_limited = np.abs(horizontal_fft_positive[limited_indices])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"FFT Analysis (fs={fs} Hz)", fontsize=16, y=0.98)

    # Plot vertical FFT
    ax1.plot(
        positive_frequencies_limited,
        vertical_magnitude_limited,
        linewidth=2.5,
        color="#1f77b4",
    )
    ax1.set_title(
        f"FFT of Vertical Signal (Centered, 0-{actual_freq_limit:.1f}Hz)",
        fontsize=14,
        pad=10,
    )
    ax1.set_xlabel("Frequency (Hz)", fontsize=12)
    ax1.set_ylabel("Magnitude", fontsize=12)
    ax1.set_xlim(0, actual_freq_limit)
    ax1.grid(True, alpha=0.3)

    # Add tight x-axis limits
    if len(positive_frequencies_limited) > 0:
        ax1.set_xlim(positive_frequencies_limited[0], positive_frequencies_limited[-1])

    # Plot horizontal FFT
    ax2.plot(
        positive_frequencies_limited,
        horizontal_magnitude_limited,
        linewidth=2.5,
        color="#ff7f0e",
    )
    ax2.set_title(
        f"FFT of Horizontal Signal (Centered, 0-{actual_freq_limit:.1f}Hz)",
        fontsize=14,
        pad=10,
    )
    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.set_ylabel("Magnitude", fontsize=12)
    ax2.set_xlim(0, actual_freq_limit)
    ax2.grid(True, alpha=0.3)

    # Add tight x-axis limits
    if len(positive_frequencies_limited) > 0:
        ax2.set_xlim(positive_frequencies_limited[0], positive_frequencies_limited[-1])

    # Add peak labels if requested
    if disp_peaks:
        # Function to group frequencies to nearest Hz and find peaks
        def find_grouped_peaks(frequencies, magnitudes, n_peaks=10, min_freq=5):
            # Filter out frequencies below min_freq
            valid_mask = frequencies > min_freq
            valid_frequencies = frequencies[valid_mask]
            valid_magnitudes = magnitudes[valid_mask]

            if len(valid_frequencies) == 0:
                return [], [], []

            # Round frequencies to nearest Hz
            rounded_freqs = np.round(valid_frequencies)

            # Create a dictionary to store the maximum magnitude for each rounded frequency
            freq_mag_dict = {}
            freq_idx_dict = {}

            for i, (freq, mag) in enumerate(zip(rounded_freqs, valid_magnitudes)):
                if freq not in freq_mag_dict or mag > freq_mag_dict[freq]:
                    freq_mag_dict[freq] = mag
                    freq_idx_dict[freq] = np.where(valid_mask)[0][
                        i
                    ]  # Get original index

            # Sort frequencies by magnitude
            sorted_freqs = sorted(
                freq_mag_dict.keys(), key=lambda x: freq_mag_dict[x], reverse=True
            )

            # Get top n frequencies
            top_freqs = sorted_freqs[:n_peaks]

            # Get the indices and magnitudes for the top frequencies
            peak_indices = [freq_idx_dict[freq] for freq in top_freqs]
            peak_magnitudes = [freq_mag_dict[freq] for freq in top_freqs]
            peak_frequencies = [frequencies[idx] for idx in peak_indices]

            return peak_indices, peak_frequencies, peak_magnitudes

        # Find and label peaks for vertical signal
        if len(vertical_magnitude_limited) > 0:
            vert_indices, vert_freqs, vert_mags = find_grouped_peaks(
                positive_frequencies_limited, vertical_magnitude_limited
            )

            for idx, freq, mag in zip(vert_indices, vert_freqs, vert_mags):
                if mag > 0:  # Only label non-zero peaks
                    ax1.plot(freq, mag, "ro", markersize=8)
                    ax1.annotate(
                        f"{freq:.1f} Hz",
                        xy=(freq, mag),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9,
                        ha="left",
                        va="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                        ),
                    )

        # Find and label peaks for horizontal signal
        if len(horizontal_magnitude_limited) > 0:
            horiz_indices, horiz_freqs, horiz_mags = find_grouped_peaks(
                positive_frequencies_limited, horizontal_magnitude_limited
            )

            for idx, freq, mag in zip(horiz_indices, horiz_freqs, horiz_mags):
                if mag > 0:  # Only label non-zero peaks
                    ax2.plot(freq, mag, "ro", markersize=8)
                    ax2.annotate(
                        f"{freq:.1f} Hz",
                        xy=(freq, mag),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9,
                        ha="left",
                        va="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                        ),
                    )

    plt.tight_layout()
    plt.close()  # Close the figure to prevent double display
    return fig


def plot_single(all_data, n_samples=5000):
    """Plot single signal with seaborn styling."""
    # Apply seaborn-style settings
    plt.style.use("seaborn")

    col1_name = all_data.columns[0]
    col2_name = all_data.columns[1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(
        all_data[col1_name].iloc[:n_samples],
        label=f"{col1_name}",
        linewidth=2,
        alpha=0.8,
    )
    axes[0].set_title(f"Signal: {col1_name}", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Samples", fontsize=12)
    axes[0].set_ylabel("Amplitude", fontsize=12)
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, linestyle="-", alpha=0.2, color="gray")

    axes[1].plot(
        all_data[col2_name].iloc[:n_samples],
        label=f"{col2_name}",
        linewidth=2,
        alpha=0.8,
    )
    axes[1].set_title(f"Signal: {col2_name}", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Samples", fontsize=12)
    axes[1].set_ylabel("Amplitude", fontsize=12)
    axes[1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, linestyle="-", alpha=0.2, color="gray")

    plt.tight_layout()
    plt.show()
    return fig


def plot_signals(
    all_data,
    vertical_raw_col="vertical_value",
    vertical_filtered_col="vertical_filtered",
    vertical_smoothed_col=None,
    horizontal_raw_col="horizontal_value",
    horizontal_filtered_col="horizontal_filtered",
    horizontal_smoothed_col=None,
    n_samples=5000,
):
    """Plot signals with seaborn styling."""
    # Apply seaborn-style settings
    plt.style.use("seaborn")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(
        all_data[vertical_raw_col].iloc[:n_samples],
        label="Original Vertical",
        linewidth=2,
        alpha=0.7,
    )
    axes[0].plot(
        all_data[vertical_filtered_col].iloc[:n_samples],
        label="Filtered Vertical",
        linewidth=2.5,
        alpha=0.9,
    )

    if vertical_smoothed_col is not None:
        axes[0].plot(
            all_data[vertical_smoothed_col].iloc[:n_samples],
            label="Smoothed Vertical",
            linewidth=2.5,
            alpha=0.9,
        )

    axes[0].set_title(
        "Vertical Signal: Original vs Filtered", fontsize=14, fontweight="bold"
    )
    axes[0].set_xlabel("Samples", fontsize=12)
    axes[0].set_ylabel("Amplitude", fontsize=12)
    axes[0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0].grid(True, linestyle="-", alpha=0.2, color="gray")

    axes[1].plot(
        all_data[horizontal_raw_col].iloc[:n_samples],
        label="Original Horizontal",
        linewidth=2,
        alpha=0.7,
    )
    axes[1].plot(
        all_data[horizontal_filtered_col].iloc[:n_samples],
        label="Filtered Horizontal",
        linewidth=2.5,
        alpha=0.9,
    )

    if horizontal_smoothed_col is not None:
        axes[1].plot(
            all_data[horizontal_smoothed_col].iloc[:n_samples],
            label="Smoothed Horizontal",
            linewidth=2.5,
            alpha=0.9,
        )

    axes[1].set_title(
        "Horizontal Signal: Original vs Filtered", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Samples", fontsize=12)
    axes[1].set_ylabel("Amplitude", fontsize=12)
    axes[1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1].grid(True, linestyle="-", alpha=0.2, color="gray")

    plt.tight_layout()
    plt.show()
    return fig


def plot_filter_response(
    sos, worN=1500, filter_type=None, cutoff_freq=None, bw=None, fs=2.0
):
    """Plot filter response with seaborn styling."""
    # Apply seaborn-style settings
    plt.style.use("seaborn")

    w, h = signal.sosfreqz(sos, worN=worN)
    w_normalized = w / np.pi
    freq = w_normalized * (fs / 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    min_db = np.floor(np.min(db) / 10) * 10
    max_db = np.ceil(np.max(db) / 10) * 10

    if max_db - min_db < 40:
        min_db = max_db - 40

    ax1.plot(freq, db, linewidth=2.5, alpha=0.9)
    ax1.set_ylim([min_db, max_db + 5])

    db_range = max_db - min_db
    if db_range <= 60:
        step = 10
    elif db_range <= 120:
        step = 20
    else:
        step = 40
    db_ticks = np.arange(min_db, max_db + step, step)
    ax1.set_yticks(db_ticks)

    if filter_type and cutoff_freq is not None:
        if filter_type.lower() in ["lowpass", "highpass"]:
            cutoff = cutoff_freq
            ax1.axvline(
                x=cutoff,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Cutoff: {cutoff:.2f} Hz",
            )
            ax1.plot(cutoff, -3, "o", color="red", markersize=8)
            ax1.annotate(
                "-3 dB",
                xy=(cutoff, -3),
                xytext=(cutoff + 0.05 * (fs / 2), -3 + 5),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#333333",
                    connectionstyle="arc3,rad=0.1",
                    shrink=0.05,
                    lw=1.5,
                ),
                fontsize=11,
                fontweight="bold",
                color="#333333",
            )
        elif filter_type.lower() in ["bandpass", "bandstop"]:
            if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
                low, high = cutoff_freq
                ax1.axvline(
                    x=low,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Lower cutoff: {low:.2f} Hz",
                )
                ax1.axvline(
                    x=high,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Upper cutoff: {high:.2f} Hz",
                )
                ax1.plot([low, high], [-3, -3], "o", color="red", markersize=8)
                ax1.annotate(
                    "-3 dB",
                    xy=(low, -3),
                    xytext=(low - 0.15 * (fs / 2), -3 + 5),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#333333",
                        connectionstyle="arc3,rad=0.1",
                        shrink=0.05,
                        lw=1.5,
                    ),
                    fontsize=11,
                    fontweight="bold",
                    color="#333333",
                )
                ax1.annotate(
                    "-3 dB",
                    xy=(high, -3),
                    xytext=(high + 0.05 * (fs / 2), -3 + 5),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#333333",
                        connectionstyle="arc3,rad=0.1",
                        shrink=0.05,
                        lw=1.5,
                    ),
                    fontsize=11,
                    fontweight="bold",
                    color="#333333",
                )
                if filter_type.lower() == "bandpass":
                    ax1.axvspan(low, high, alpha=0.15, color="green")
                else:
                    ax1.axvspan(0, low, alpha=0.15, color="green")
                    ax1.axvspan(high, fs / 2, alpha=0.15, color="green")
            else:
                print(
                    "For bandpass/bandstop filters, provide cutoff_freq as [low, high]"
                )

    ax1.grid(True, which="both", linestyle="-", alpha=0.2, color="gray")
    ax1.set_ylabel("Magnitude [dB]", fontsize=12)
    if fs == 2.0:
        ax1.set_xlabel("Normalized frequency (1.0 = Nyquist)", fontsize=12)
    else:
        ax1.set_xlabel("Frequency [Hz]", fontsize=12)

    title = "Filter Frequency Response"
    if filter_type:
        title = f"{filter_type.capitalize()} Filter Frequency Response"
        if filter_type.lower() in ["lowpass", "highpass"] and cutoff_freq is not None:
            title += f" (Cutoff: {cutoff_freq:.2f} Hz)"
        elif filter_type.lower() in ["bandpass", "bandstop"] and isinstance(
            cutoff_freq, (list, tuple)
        ):
            title += f" (Cutoffs: {cutoff_freq[0]:.2f}-{cutoff_freq[1]:.2f} Hz)"
    ax1.set_title(title, fontsize=14, fontweight="bold")
    if filter_type:
        ax1.legend(loc="best", frameon=True, fancybox=True, shadow=True)

    phase = np.unwrap(np.angle(h))
    ax2.plot(freq, phase, linewidth=2.5, alpha=0.9)
    ax2.grid(True, which="both", linestyle="-", alpha=0.2, color="gray")
    ax2.set_ylabel("Phase [rad]", fontsize=12)
    if fs == 2.0:
        ax2.set_xlabel("Normalized frequency (1.0 = Nyquist)", fontsize=12)
    else:
        ax2.set_xlabel("Frequency [Hz]", fontsize=12)

    phase_ticks = [-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi, np.pi]
    phase_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    ax2.set_yticks(phase_ticks)
    ax2.set_yticklabels(phase_labels)

    x_limits = [0, fs / 2]
    ax1.set_xlim(x_limits)
    ax2.set_xlim(x_limits)

    plt.tight_layout()
    return fig


def get_csv_for_participant(participant_id: int, dir):
    files = os.listdir(dir)
    filtered_files = [
        file
        for file in files
        if file.endswith(".csv") and f"participant_{participant_id}" in file
    ]
    sorted_files = sorted(
        filtered_files, key=lambda x: int(x.split("_")[1].replace("session", ""))
    )
    return sorted_files


def get_data_for_participant(participant: str, base_path: str):
    """
    Reads, cleans, resamples, and concatenates participant data from CSV files.
    Gaps between files are NOT filled.
    """

    default_fill_value = 2.5
    final_df = None
    processed_a_file = False
    cols_order = None

    # Sort files by name assumes chronological order if filenames reflect that.
    # If order is critical and not guaranteed by filenames, adjust file fetching/sorting.
    files = sorted(get_csv_for_participant(participant, base_path))

    for file in files:
        df = pd.read_csv(base_path + file)
        df["corrected_timestamp"] = pd.to_datetime(df["timestamp"] + 3600, unit="s")
        df = df.drop_duplicates(subset=["corrected_timestamp"], keep="first")
        df.sort_values("corrected_timestamp", inplace=True)

        # --- Frequency Deduction ---
        file_start_ts = df["corrected_timestamp"].iloc[0]
        file_end_ts = df["corrected_timestamp"].iloc[-1]
        file_points = len(df)
        time_span_seconds = (file_end_ts - file_start_ts).total_seconds()

        # Happy path: file_points > 1 and time_span_seconds > 0
        freq_hz = (file_points - 1) / time_span_seconds
        if freq_hz >= 1:
            frequency = f"{int(round(1000/freq_hz))}ms"  # Milliseconds or faster
        else:
            frequency = f"{int(round(1/freq_hz))}s"  # Seconds
        freq_delta = pd.Timedelta(frequency)

        # --- Reindexing ---
        ideal_index = pd.date_range(
            start=file_start_ts, periods=file_points, freq=freq_delta
        )

        df.set_index("corrected_timestamp", inplace=True)
        current_cols = df.columns.drop("timestamp", errors="ignore")
        if cols_order is None:
            cols_order = current_cols  # Capture column order from the first file

        reindexed_df = df[current_cols].reindex(ideal_index, method="nearest")
        reindexed_df.index.name = "timestamp"
        reindexed_df.attrs["freq_delta"] = freq_delta  # Store only delta needed later

        # --- Concatenation & Gap Filling ---
        if not processed_a_file:
            final_df = reindexed_df
            processed_a_file = True
        else:
            last_timestamp_prev = final_df.index.max()
            first_timestamp_next = reindexed_df.index.min()
            freq_delta_prev = final_df.attrs[
                "freq_delta"
            ]  # Use previous segment's freq

            gap_start_time = last_timestamp_prev + freq_delta_prev
            gap_end_time_calc = first_timestamp_next - freq_delta_prev

            # Minimal check to prevent invalid date_range or empty DataFrames
            if gap_start_time < first_timestamp_next:
                gap_index = pd.date_range(
                    start=gap_start_time, end=gap_end_time_calc, freq=freq_delta_prev
                )
                if not gap_index.empty:
                    gap_df = pd.DataFrame(
                        default_fill_value, index=gap_index, columns=cols_order
                    )
                    final_df = pd.concat([final_df, gap_df])

            final_df = pd.concat([final_df, reindexed_df])
            # Update final_df attrs with the latest segment's frequency delta
            final_df.attrs["freq_delta"] = reindexed_df.attrs["freq_delta"]

    return final_df


def signal_processing(participant: int, data_path="data/"):
    df_final = get_data_for_participant(1, data_path + "raw/")

    all_data = df_final.resample("1ms").nearest(limit=2).fillna(2.5)
    fs = 1 / pd.Series(all_data.index.diff().total_seconds()).median()
    filter_order = 4

    sos = signal.butter(
        filter_order, 7.5, btype="lowpass", analog=False, output="sos", fs=fs
    )

    all_data["vertical_filtered"] = signal.sosfilt(sos, all_data["vertical_value"])
    all_data["horizontal_filtered"] = signal.sosfilt(sos, all_data["horizontal_value"])

    all_data = all_data.resample("5ms").mean()

    lowcut_freq = 0.5
    highcut_freq = 7.5
    filter_order = 5

    sos = signal.butter(
        filter_order,
        [lowcut_freq, highcut_freq],
        btype="band",
        analog=False,
        output="sos",
        fs=fs,
    )

    all_data["vertical_filtered"] = signal.sosfiltfilt(sos, all_data["vertical_value"])
    all_data["horizontal_filtered"] = signal.sosfiltfilt(
        sos, all_data["horizontal_value"]
    )

    all_data.to_parquet(data_path + f"{participant}_processed_data.parquet")


if __name__ == "__main__":
    signal_processing(1)
