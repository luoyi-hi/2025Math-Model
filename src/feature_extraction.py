"""Feature extraction pipeline for the bearing transfer learning task.

This script prepares a curated source-domain dataset using the provided CWRU
bearing data. It performs the following steps:

1. Collect the 12 kHz drive-end fault signals (rolling element, inner race and
   centered outer race faults) and the normal-condition baselines.
2. Optionally resample signals to a common rate and clip them to a fixed
   duration to match the target-domain recording setup.
3. Compute a rich set of time-domain, frequency-domain and envelope features for
   each signal to characterise the bearing health state.
4. Write the per-record features and aggregated summaries to CSV files, which
   can be used in downstream transfer-learning experiments.

Usage
-----

```bash
python -m src.feature_extraction \
    --source-root 数据集/源域数据集 \
    --output-dir artifacts/source_features \
    --sensor DE \
    --target-sample-rate 12000 \
    --clip-seconds 8
```

The output directory will contain three files:
```
- features.csv:       per-record feature table
- feature_means.csv:  class-wise mean feature values
- dataset_summary.json: metadata about the curated dataset
```
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert, resample
from scipy.stats import entropy, kurtosis, skew


@dataclass
class BearingRecord:
    """Container for metadata describing a single vibration record."""

    path: Path
    sensor: str
    fault_type: str
    fault_size: Optional[float]
    load: Optional[int]
    orientation: Optional[str]
    rpm: Optional[float]
    sample_rate: float

    def to_dict(self) -> Dict[str, Optional[float]]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def _parse_fault_metadata(path: Path, sensor: str) -> Tuple[str, Optional[float], Optional[int], Optional[str]]:
    """Infer fault attributes from the file path."""

    stem = path.stem
    parts = path.parts
    orientation = None

    if "OR" in parts:
        # Orientation stored two levels above file (Centered/Opposite/Orthogonal).
        try:
            idx = parts.index("OR")
            orientation = parts[idx + 1]
        except ValueError:
            orientation = None

    load_match = re.search(r"_(\d)(?:_|$)", stem)
    load = int(load_match.group(1)) if load_match else None

    if stem.startswith("B"):
        fault_type = "B"
        severity = float(stem[1:4]) / 1000.0
    elif stem.startswith("IR"):
        fault_type = "IR"
        severity = float(stem[2:5]) / 1000.0
    elif stem.startswith("OR"):
        fault_type = "OR"
        severity = float(stem[2:5]) / 1000.0
        if orientation is None:
            # Fallback for safety.
            orientation = "Unknown"
    elif stem.startswith("N"):
        fault_type = "N"
        severity = None
        if load is None:
            load_fragments = re.findall(r"_(\d)", stem)
            load = int(load_fragments[0]) if load_fragments else None
    else:
        raise ValueError(f"Unrecognised file naming scheme for {path}")

    return fault_type, severity, load, orientation


def _extract_signal(mat: Dict[str, np.ndarray], sensor: str) -> np.ndarray:
    """Fetch the vibration signal for the requested sensor."""

    suffix = f"_{sensor}_time"
    for key, value in mat.items():
        if key.endswith(suffix):
            return np.asarray(value).squeeze()
    raise KeyError(f"No {sensor} channel found in keys: {list(mat)}")


def _extract_rpm(mat: Dict[str, np.ndarray]) -> Optional[float]:
    for key, value in mat.items():
        if key.lower().endswith("rpm"):
            arr = np.asarray(value).squeeze()
            if arr.size == 0:
                continue
            return float(np.mean(arr))
    return None


def collect_drive_end_records(root: Path, *, orientation: str = "Centered") -> List[BearingRecord]:
    """Collect 12 kHz drive-end fault records plus baseline normal samples."""

    records: List[BearingRecord] = []
    sensor = "DE"
    base_dir = root / "12kHz_DE_data"
    sample_rate = 12000.0

    if not base_dir.exists():
        raise FileNotFoundError(f"Expected directory missing: {base_dir}")

    # Rolling element and inner race faults.
    for fault_group in ["B", "IR"]:
        fault_dir = base_dir / fault_group
        for severity_dir in sorted(fault_dir.iterdir()):
            if not severity_dir.is_dir():
                continue
            for mat_path in sorted(severity_dir.glob("*.mat")):
                fault_type, severity, load, _ = _parse_fault_metadata(mat_path, sensor)
                records.append(
                    BearingRecord(
                        path=mat_path,
                        sensor=sensor,
                        fault_type=fault_type,
                        fault_size=severity,
                        load=load,
                        orientation=None,
                        rpm=None,
                        sample_rate=sample_rate,
                    )
                )

    # Outer race faults - use specified orientation for consistency.
    or_dir = base_dir / "OR" / orientation
    if not or_dir.exists():
        raise FileNotFoundError(f"Expected outer-race orientation directory missing: {or_dir}")

    for severity_dir in sorted(or_dir.iterdir()):
        if not severity_dir.is_dir():
            continue
        for mat_path in sorted(severity_dir.glob("*.mat")):
            fault_type, severity, load, orient = _parse_fault_metadata(mat_path, sensor)
            records.append(
                BearingRecord(
                    path=mat_path,
                    sensor=sensor,
                    fault_type=fault_type,
                    fault_size=severity,
                    load=load,
                    orientation=orient,
                    rpm=None,
                    sample_rate=sample_rate,
                )
            )

    # Normal baselines (48 kHz) - resample later to align with 12 kHz faults.
    normal_dir = root / "48kHz_Normal_data"
    if normal_dir.exists():
        for mat_path in sorted(normal_dir.glob("*.mat")):
            fault_type, severity, load, orientation = _parse_fault_metadata(mat_path, sensor)
            records.append(
                BearingRecord(
                    path=mat_path,
                    sensor=sensor,
                    fault_type=fault_type,
                    fault_size=severity,
                    load=load,
                    orientation=orientation,
                    rpm=None,
                    sample_rate=48000.0,
                )
            )
    else:
        raise FileNotFoundError(f"Normal-condition directory missing: {normal_dir}")

    return records


def _prepare_signal(
    signal: np.ndarray,
    *,
    original_rate: float,
    target_rate: Optional[float],
    clip_seconds: Optional[float],
) -> Tuple[np.ndarray, float]:
    """Resample and clip the vibration signal."""

    if signal.ndim != 1:
        signal = signal.reshape(-1)

    effective_rate = original_rate
    if target_rate is not None and not math.isclose(original_rate, target_rate):
        # Resample to target rate.
        target_length = int(round(len(signal) * target_rate / original_rate))
        signal = resample(signal, target_length)
        effective_rate = target_rate

    if clip_seconds is not None and clip_seconds > 0:
        target_length = int(round(effective_rate * clip_seconds))
        if target_length <= 0:
            raise ValueError("clip_seconds resulted in non-positive target length")
        if len(signal) >= target_length:
            signal = signal[:target_length]
        else:
            padding = target_length - len(signal)
            signal = np.pad(signal, (0, padding), mode="edge")

    return signal.astype(np.float64), effective_rate


def _time_features(signal: np.ndarray) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if signal.size == 0:
        raise ValueError("Empty signal encountered during feature computation")

    abs_signal = np.abs(signal)
    mean_val = float(np.mean(signal))
    rms_val = float(np.sqrt(np.mean(signal ** 2)))
    std_val = float(np.std(signal))
    abs_mean = float(np.mean(abs_signal))
    square_mean = float(np.mean(signal ** 2))
    peak_val = float(np.max(abs_signal))
    peak_to_peak = float(np.max(signal) - np.min(signal))
    crest_factor = peak_val / rms_val if rms_val > 0 else 0.0
    impulse_factor = peak_val / abs_mean if abs_mean > 0 else 0.0
    clearance_den = np.mean(np.sqrt(abs_signal))
    clearance_factor = peak_val / (clearance_den ** 2) if clearance_den > 0 else 0.0
    shape_factor = rms_val / abs_mean if abs_mean > 0 else 0.0
    kurtosis_factor = float(np.mean(signal ** 4) / (square_mean ** 2)) if square_mean > 0 else 0.0
    skewness = float(skew(signal))
    kurt_val = float(kurtosis(signal, fisher=True))
    entropy_val = float(entropy(np.histogram(signal, bins=64, density=True)[0] + 1e-12))
    zero_crossings = float(np.sum(signal[:-1] * signal[1:] < 0))
    zcr = zero_crossings / (signal.size - 1)

    features.update(
        {
            "mean": mean_val,
            "std": std_val,
            "rms": rms_val,
            "variance": float(np.var(signal)),
            "abs_mean": abs_mean,
            "peak": peak_val,
            "peak_to_peak": peak_to_peak,
            "crest_factor": crest_factor,
            "impulse_factor": impulse_factor,
            "clearance_factor": clearance_factor,
            "shape_factor": shape_factor,
            "kurtosis_factor": kurtosis_factor,
            "skewness": skewness,
            "kurtosis": kurt_val,
            "entropy": entropy_val,
            "zero_crossing_rate": zcr,
            "signal_energy": float(np.sum(signal ** 2)),
        }
    )

    return features


def _frequency_features(
    signal: np.ndarray,
    sample_rate: float,
    *,
    rpm: Optional[float] = None,
    band_edges: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    if band_edges is None:
        nyquist = sample_rate / 2.0
        band_edges = [0.0, 300.0, 600.0, 1200.0, 2400.0, nyquist]
    else:
        band_edges = list(band_edges)

    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    magnitudes = np.abs(fft_vals)
    power = magnitudes ** 2
    total_power = np.sum(power)
    if total_power <= 0:
        total_power = 1e-12

    centroid = float(np.sum(freqs * power) / total_power)
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    spectral_entropy = float(entropy(power / total_power))
    dominant_idx = int(np.argmax(power))
    dominant_freq = float(freqs[dominant_idx])
    bandwidth = float(np.sqrt(np.sum(((freqs - dominant_freq) ** 2) * power) / total_power))

    features: Dict[str, float] = {
        "spectral_centroid": centroid,
        "spectral_spread": spread,
        "spectral_entropy": spectral_entropy,
        "dominant_frequency": dominant_freq,
        "spectral_bandwidth": bandwidth,
    }

    # Band-specific energy ratios.
    band_edges = sorted(band_edges)
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        if high <= low:
            continue
        mask = (freqs >= low) & (freqs < high)
        band_power = float(np.sum(power[mask]))
        label = f"band_energy_{int(low)}_{int(high)}"
        features[label] = band_power / total_power

    # Cumulative energy up to characteristic harmonics (1x and 2x shaft speed).
    shaft_freq = None if rpm is None or np.isnan(rpm) else rpm / 60.0
    if shaft_freq is not None and shaft_freq > 0:
        for multiple in [1, 2, 3]:
            upper = shaft_freq * (multiple + 0.5)
            mask = freqs <= upper
            features[f"cum_energy_upto_{multiple}x"] = float(np.sum(power[mask]) / total_power)

    # Spectral skewness and kurtosis using power distribution.
    norm_power = power / total_power
    features["spectral_skewness"] = float(np.sum(((freqs - centroid) ** 3) * norm_power) / (spread ** 3 + 1e-12))
    features["spectral_kurtosis"] = float(np.sum(((freqs - centroid) ** 4) * norm_power) / (spread ** 4 + 1e-12))

    return features


def _envelope_features(signal: np.ndarray, sample_rate: float) -> Dict[str, float]:
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    envelope_fft = np.fft.rfft(envelope)
    freqs = np.fft.rfftfreq(envelope.size, d=1.0 / sample_rate)
    power = np.abs(envelope_fft) ** 2
    total_power = np.sum(power)
    if total_power <= 0:
        total_power = 1e-12

    mean_env = float(np.mean(envelope))
    std_env = float(np.std(envelope))
    peak_env = float(np.max(envelope))
    crest_env = peak_env / (np.sqrt(np.mean(envelope ** 2)) + 1e-12)
    env_centroid = float(np.sum(freqs * power) / total_power)

    return {
        "envelope_mean": mean_env,
        "envelope_std": std_env,
        "envelope_peak": peak_env,
        "envelope_crest_factor": crest_env,
        "envelope_spectral_centroid": env_centroid,
    }


def extract_features_for_record(
    record: BearingRecord,
    *,
    target_sample_rate: Optional[float] = None,
    clip_seconds: Optional[float] = None,
) -> Dict[str, float]:
    mat = loadmat(record.path)
    signal = _extract_signal(mat, record.sensor)
    rpm = _extract_rpm(mat)

    prepared_signal, effective_rate = _prepare_signal(
        signal,
        original_rate=record.sample_rate,
        target_rate=target_sample_rate,
        clip_seconds=clip_seconds,
    )

    features = {
        "file_path": str(record.path),
        "sensor": record.sensor,
        "fault_type": record.fault_type,
        "fault_size": record.fault_size if record.fault_size is not None else np.nan,
        "load": record.load if record.load is not None else np.nan,
        "orientation": record.orientation if record.orientation is not None else "",
        "rpm": rpm if rpm is not None else np.nan,
        "sample_rate": effective_rate,
        "signal_length": prepared_signal.size,
    }

    features.update(_time_features(prepared_signal))
    freq_features = _frequency_features(prepared_signal, effective_rate, rpm=features["rpm"])
    features.update(freq_features)
    features.update(_envelope_features(prepared_signal, effective_rate))

    return features


def _summarise_dataset(records: List[BearingRecord], features_df: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "num_records": len(records),
        "fault_type_counts": features_df["fault_type"].value_counts().to_dict(),
        "loads": sorted(features_df["load"].dropna().unique().tolist()),
        "sample_rate_distribution": features_df["sample_rate"].value_counts().to_dict(),
    }
    if "fault_size" in features_df.columns:
        summary["fault_sizes"] = sorted(features_df["fault_size"].dropna().unique().tolist())
    return summary


def run_pipeline(
    source_root: Path,
    output_dir: Path,
    *,
    sensor: str,
    orientation: str,
    target_sample_rate: Optional[float],
    clip_seconds: Optional[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if sensor.upper() != "DE":
        raise NotImplementedError("Currently only drive-end sensor data is curated")

    records = collect_drive_end_records(source_root, orientation=orientation)

    feature_rows = []
    for record in records:
        feature_row = extract_features_for_record(
            record,
            target_sample_rate=target_sample_rate,
            clip_seconds=clip_seconds,
        )
        feature_rows.append(feature_row)

    features_df = pd.DataFrame(feature_rows)
    features_path = output_dir / "features.csv"
    features_df.to_csv(features_path, index=False)

    # Class-wise mean feature values to support quick analysis.
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    mean_df = features_df.groupby("fault_type")[numeric_cols].mean().reset_index()
    mean_path = output_dir / "feature_means.csv"
    mean_df.to_csv(mean_path, index=False)

    summary = _summarise_dataset(records, features_df)
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"Wrote {len(features_df)} feature rows to {features_path}")
    print(f"Class-wise statistics saved to {mean_path}")
    print(f"Dataset summary saved to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bearing feature extraction pipeline")
    parser.add_argument("--source-root", type=Path, default=Path("数据集/源域数据集"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/source_features"))
    parser.add_argument("--sensor", type=str, default="DE", help="Sensor channel to use (currently only DE supported)")
    parser.add_argument(
        "--orientation",
        type=str,
        default="Centered",
        help="Outer-race fault orientation to include (Centered/Opposite/Orthogonal)",
    )
    parser.add_argument("--target-sample-rate", type=float, default=12000.0, help="Resample signals to this rate")
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=8.0,
        help="Clip or pad each signal to this duration in seconds (use 0 to disable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_seconds = None if args.clip_seconds is None or args.clip_seconds <= 0 else args.clip_seconds
    run_pipeline(
        source_root=args.source_root,
        output_dir=args.output_dir,
        sensor=args.sensor,
        orientation=args.orientation,
        target_sample_rate=args.target_sample_rate,
        clip_seconds=clip_seconds,
    )


if __name__ == "__main__":
    main()
