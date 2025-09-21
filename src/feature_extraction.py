"""Comprehensive feature-extraction pipeline for the bearing transfer-learning task.

This script processes every MATLAB record inside the provided source and target
vibration datasets.  It consolidates metadata, optionally resamples each signal
so that all records share a common sampling rate, extracts rich diagnostic
features, and exports analysis-ready tables for downstream modelling.

Key capabilities
----------------
* Traverse the full directory tree under ``数据集/源域数据集`` (12 kHz/48 kHz
  drive-end, fan-end and normal conditions) and ``数据集/目标域数据集`` (32 kHz
  train-borne measurements).
* Infer fault type, severity, load condition and orientation directly from file
  names while gracefully handling unlabeled target-domain records.
* Compute comprehensive time, frequency and envelope indicators for every
  record, providing a consistent feature representation for transfer-learning
  experiments.
* Emit metadata catalogues, per-record feature tables, aggregated statistics and
  dataset summaries to facilitate exploratory analysis and model development.

Example
-------
The following command processes both source and target datasets, resamples all
signals to 12 kHz, clips/pads them to 8 seconds and writes the resulting
artifacts to ``artifacts/full_features``::

    python -m src.feature_extraction \
        --source-root 数据集/源域数据集 \
        --target-root 数据集/目标域数据集 \
        --output-dir artifacts/full_features \
        --target-sample-rate 12000 \
        --clip-seconds 8

Outputs
-------
``record_catalog.csv``
    Metadata describing each processed file (domain, dataset, sensor, label,
    etc.).
``features.csv``
    Per-record diagnostic features that can be fed into machine-learning
    pipelines.
``feature_means.csv``
    Aggregated feature means grouped by domain, dataset, sensor and fault type.
``dataset_summary.json``
    High-level overview of the curated dataset (record counts, label
    distribution, sampling rates, durations, etc.).
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


_ORIENTATIONS = {"Centered", "Opposite", "Orthogonal"}


@dataclass
class BearingRecord:
    """Container describing a single vibration record to be processed."""

    path: Path
    domain: str
    dataset: str
    sensor: Optional[str]
    fault_type: str
    fault_size: Optional[float]
    load: Optional[int]
    orientation: Optional[str]
    sample_rate: float

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["sensor"] = data["sensor"] if data["sensor"] is not None else "Unknown"
        data["orientation"] = data["orientation"] if data["orientation"] is not None else ""
        return data


def _parse_source_top_dir(component: str) -> Tuple[float, Optional[str]]:
    """Infer the sampling rate and sensor family from a directory name."""

    match = re.match(r"(?P<rate>\d+)kHz_(?P<label>.+)_data$", component)
    if not match:
        raise ValueError(f"Unrecognised source dataset directory: {component}")

    sample_rate = float(match.group("rate")) * 1000.0
    label = match.group("label").upper()

    if label in {"DE", "FE", "BA"}:
        sensor = label
    elif label == "NORMAL":
        sensor = "DE"
    else:
        sensor = None

    return sample_rate, sensor


def _parse_fault_metadata(path: Path) -> Tuple[str, Optional[float], Optional[int], Optional[str]]:
    """Infer fault attributes from the file name and directory structure."""

    stem = path.stem
    orientation = None
    for part in path.parts:
        if part in _ORIENTATIONS:
            orientation = part
            break

    load_match = re.search(r"_(\d)(?:_|$)", stem)
    load = int(load_match.group(1)) if load_match else None

    if stem.startswith("B") and len(stem) >= 4:
        fault_type = "B"
        severity = float(stem[1:4]) / 1000.0
    elif stem.startswith("IR") and len(stem) >= 5:
        fault_type = "IR"
        severity = float(stem[2:5]) / 1000.0
    elif stem.startswith("OR") and len(stem) >= 5:
        fault_type = "OR"
        severity = float(stem[2:5]) / 1000.0
        if orientation is None:
            orientation = "Unknown"
    elif stem.startswith("N"):
        fault_type = "N"
        severity = None
        if load is None:
            load_digits = re.findall(r"_(\d)", stem)
            load = int(load_digits[0]) if load_digits else None
    else:
        raise ValueError(f"Unrecognised file naming scheme for {path}")

    return fault_type, severity, load, orientation


def collect_source_records(
    root: Path,
    *,
    include_datasets: Optional[Sequence[str]] = None,
    include_sensors: Optional[Sequence[str]] = None,
) -> List[BearingRecord]:
    """Scan the source-domain tree and build a catalogue of records."""

    if not root.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {root}")

    dataset_filter = set(include_datasets) if include_datasets else None
    sensor_filter = {s.upper() for s in include_sensors} if include_sensors else None

    records: List[BearingRecord] = []
    for mat_path in sorted(root.rglob("*.mat")):
        relative = mat_path.relative_to(root)
        top_component = relative.parts[0]
        if dataset_filter and top_component not in dataset_filter:
            continue

        sample_rate, sensor = _parse_source_top_dir(top_component)
        if sensor_filter and (sensor is None or sensor not in sensor_filter):
            continue

        fault_type, severity, load, orientation = _parse_fault_metadata(mat_path)

        records.append(
            BearingRecord(
                path=mat_path,
                domain="source",
                dataset=top_component,
                sensor=sensor,
                fault_type=fault_type,
                fault_size=severity,
                load=load,
                orientation=orientation,
                sample_rate=sample_rate,
            )
        )

    return records


def collect_target_records(root: Path, *, sample_rate: float = 32000.0) -> List[BearingRecord]:
    """Create records for every target-domain measurement."""

    if not root.exists():
        raise FileNotFoundError(f"Target dataset directory not found: {root}")

    records: List[BearingRecord] = []
    for mat_path in sorted(root.rglob("*.mat")):
        records.append(
            BearingRecord(
                path=mat_path,
                domain="target",
                dataset=root.name,
                sensor=None,
                fault_type="Unknown",
                fault_size=None,
                load=None,
                orientation=None,
                sample_rate=sample_rate,
            )
        )

    return records


def _extract_signal(
    mat: Dict[str, np.ndarray],
    *,
    record: BearingRecord,
) -> Tuple[np.ndarray, str]:
    """Fetch the vibration signal array and the key it originated from."""

    def _is_valid_key(key: str) -> bool:
        if key.startswith("__"):
            return False
        if key.lower().endswith("rpm"):
            return False
        return True

    def _load_if_valid(key: str) -> Optional[np.ndarray]:
        if not _is_valid_key(key):
            return None
        array = np.asarray(mat[key]).squeeze()
        if array.ndim != 1 or array.size <= 1:
            return None
        return array

    if record.sensor:
        suffix = f"_{record.sensor}_time"
        for key in mat:
            if key.endswith(suffix):
                array = _load_if_valid(key)
                if array is not None:
                    return array, key

    stem = record.path.stem
    if stem in mat:
        array = _load_if_valid(stem)
        if array is not None:
            return array, stem

    for key in mat:
        array = _load_if_valid(key)
        if array is not None:
            return array, key

    raise KeyError(f"No usable signal channel found in {record.path}")


def _extract_rpm(mat: Dict[str, np.ndarray]) -> Optional[float]:
    for key, value in mat.items():
        if key.lower().endswith("rpm"):
            arr = np.asarray(value).squeeze()
            if arr.size == 0:
                continue
            return float(np.mean(arr))
    return None


def _prepare_signal(
    signal: np.ndarray,
    *,
    original_rate: float,
    target_rate: Optional[float],
    clip_seconds: Optional[float],
) -> Tuple[np.ndarray, float]:
    """Resample and optionally clip/pad a signal."""

    if signal.ndim != 1:
        signal = signal.reshape(-1)

    effective_rate = original_rate
    if target_rate is not None and target_rate > 0 and not math.isclose(original_rate, target_rate):
        target_length = int(round(len(signal) * target_rate / original_rate))
        target_length = max(target_length, 1)
        signal = resample(signal, target_length)
        effective_rate = target_rate

    if clip_seconds is not None and clip_seconds > 0:
        target_length = int(round(effective_rate * clip_seconds))
        target_length = max(target_length, 1)
        if len(signal) >= target_length:
            signal = signal[:target_length]
        else:
            padding = target_length - len(signal)
            signal = np.pad(signal, (0, padding), mode="edge")

    return signal.astype(np.float64), effective_rate


def _time_features(signal: np.ndarray) -> Dict[str, float]:
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
    histogram = np.histogram(signal, bins=64, density=True)[0] + 1e-12
    entropy_val = float(entropy(histogram))
    zero_crossings = float(np.sum(signal[:-1] * signal[1:] < 0))
    zcr = zero_crossings / max(signal.size - 1, 1)

    return {
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

    band_edges = sorted(band_edges)
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        if high <= low:
            continue
        mask = (freqs >= low) & (freqs < high)
        band_power = float(np.sum(power[mask]))
        label = f"band_energy_{int(low)}_{int(high)}"
        features[label] = band_power / total_power

    shaft_freq = None if rpm is None or np.isnan(rpm) else rpm / 60.0
    if shaft_freq is not None and shaft_freq > 0:
        for multiple in [1, 2, 3]:
            upper = shaft_freq * (multiple + 0.5)
            mask = freqs <= upper
            features[f"cum_energy_upto_{multiple}x"] = float(np.sum(power[mask]) / total_power)

    norm_power = power / total_power
    spread_safe = spread if spread > 1e-12 else 1e-12
    features["spectral_skewness"] = float(
        np.sum(((freqs - centroid) ** 3) * norm_power) / (spread_safe ** 3)
    )
    features["spectral_kurtosis"] = float(
        np.sum(((freqs - centroid) ** 4) * norm_power) / (spread_safe ** 4)
    )

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
    signal, channel_key = _extract_signal(mat, record=record)
    rpm = _extract_rpm(mat)

    prepared_signal, effective_rate = _prepare_signal(
        signal,
        original_rate=record.sample_rate,
        target_rate=target_sample_rate,
        clip_seconds=clip_seconds,
    )

    duration = prepared_signal.size / effective_rate if effective_rate > 0 else float("nan")
    sensor_label = record.sensor if record.sensor is not None else "Unknown"

    features: Dict[str, float] = {
        "domain": record.domain,
        "dataset": record.dataset,
        "file_path": str(record.path),
        "file_name": record.path.name,
        "sensor": sensor_label,
        "channel_key": channel_key,
        "fault_type": record.fault_type,
        "fault_size": record.fault_size if record.fault_size is not None else np.nan,
        "load": record.load if record.load is not None else np.nan,
        "orientation": record.orientation if record.orientation is not None else "",
        "rpm": rpm if rpm is not None else np.nan,
        "sample_rate": effective_rate,
        "original_sample_rate": record.sample_rate,
        "resample_factor": effective_rate / record.sample_rate if record.sample_rate > 0 else np.nan,
        "signal_length": prepared_signal.size,
        "duration_seconds": duration,
    }

    features.update(_time_features(prepared_signal))
    freq_features = _frequency_features(prepared_signal, effective_rate, rpm=features["rpm"])
    features.update(freq_features)
    features.update(_envelope_features(prepared_signal, effective_rate))

    return features


def _summarise_dataset(records: Sequence[BearingRecord], features_df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "num_records": int(len(records)),
        "domains": {k: int(v) for k, v in features_df["domain"].value_counts().items()},
        "datasets": {k: int(v) for k, v in features_df["dataset"].value_counts().items()},
        "sensors": {k: int(v) for k, v in features_df["sensor"].value_counts().items()},
        "fault_type_counts": {k: int(v) for k, v in features_df["fault_type"].value_counts().items()},
    }

    domain_fault_counts: Dict[str, Dict[str, int]] = {}
    for domain, group in features_df.groupby("domain"):
        domain_fault_counts[domain] = {k: int(v) for k, v in group["fault_type"].value_counts().items()}
    summary["fault_type_counts_by_domain"] = domain_fault_counts

    loads = features_df["load"].dropna()
    if not loads.empty:
        summary["loads"] = sorted(int(x) for x in loads.unique())

    fault_sizes = features_df["fault_size"].dropna()
    if not fault_sizes.empty:
        summary["fault_sizes"] = sorted(float(x) for x in fault_sizes.unique())

    summary["original_sample_rate_distribution"] = {
        float(rate): int(count)
        for rate, count in features_df["original_sample_rate"].value_counts().items()
    }
    summary["effective_sample_rate_distribution"] = {
        float(rate): int(count)
        for rate, count in features_df["sample_rate"].value_counts().items()
    }

    durations = features_df["duration_seconds"].dropna()
    if not durations.empty:
        summary["duration_seconds"] = {
            "min": float(durations.min()),
            "max": float(durations.max()),
            "mean": float(durations.mean()),
        }

    summary["example_records"] = {
        domain: group["file_name"].drop_duplicates().sort_values().head(5).tolist()
        for domain, group in features_df.groupby("domain")
    }

    return summary


def run_pipeline(
    *,
    source_root: Path,
    target_root: Optional[Path],
    output_dir: Path,
    include_source: bool,
    include_target: bool,
    include_sensors: Optional[Sequence[str]],
    include_datasets: Optional[Sequence[str]],
    target_sample_rate: Optional[float],
    clip_seconds: Optional[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[BearingRecord] = []

    sensor_filter = {s.upper() for s in include_sensors} if include_sensors else None

    if include_source:
        records.extend(
            collect_source_records(
                source_root,
                include_datasets=include_datasets,
                include_sensors=include_sensors,
            )
        )

    if include_target and target_root is not None:
        records.extend(collect_target_records(target_root))

    if sensor_filter:
        filtered_records = []
        for record in records:
            label = record.sensor.upper() if record.sensor else "UNKNOWN"
            if label in sensor_filter:
                filtered_records.append(record)
        records = filtered_records

    if not records:
        raise RuntimeError("No records were collected. Check dataset paths and filters.")

    feature_rows = []
    for record in records:
        feature_row = extract_features_for_record(
            record,
            target_sample_rate=target_sample_rate,
            clip_seconds=clip_seconds,
        )
        feature_rows.append(feature_row)

    features_df = pd.DataFrame(feature_rows)
    features_df.sort_values(["domain", "dataset", "file_name"], inplace=True)

    catalog_path = output_dir / "record_catalog.csv"
    catalog_df = pd.DataFrame([record.to_dict() for record in records])
    catalog_df.sort_values(["domain", "dataset", "path"], inplace=True)
    catalog_df.to_csv(catalog_path, index=False)

    features_path = output_dir / "features.csv"
    features_df.to_csv(features_path, index=False)

    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    group_cols = ["domain", "dataset", "sensor", "fault_type"]
    mean_df = features_df.groupby(group_cols)[numeric_cols].mean().reset_index()
    mean_path = output_dir / "feature_means.csv"
    mean_df.to_csv(mean_path, index=False)

    summary = _summarise_dataset(records, features_df)
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"Processed {len(records)} records. Metadata saved to {catalog_path}")
    print(f"Wrote per-record features to {features_path}")
    print(f"Class-wise statistics saved to {mean_path}")
    print(f"Dataset summary saved to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bearing feature extraction pipeline")
    parser.add_argument("--source-root", type=Path, default=Path("数据集/源域数据集"))
    parser.add_argument("--target-root", type=Path, default=Path("数据集/目标域数据集"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/full_features"))
    parser.add_argument("--skip-source", action="store_true", help="Do not process source-domain records")
    parser.add_argument("--skip-target", action="store_true", help="Do not process target-domain records")
    parser.add_argument(
        "--sensors",
        nargs="*",
        default=None,
        help="Optional list of sensors to include (e.g. DE FE BA Unknown). Defaults to all",
    )
    parser.add_argument(
        "--source-datasets",
        nargs="*",
        default=None,
        help="Optional list of source dataset directory names to include",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=float,
        default=12000.0,
        help="Resample every signal to this rate (<=0 keeps the original rate)",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=8.0,
        help="Clip or pad signals to this duration in seconds (<=0 disables clipping)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_rate = args.target_sample_rate if args.target_sample_rate and args.target_sample_rate > 0 else None
    clip_seconds = args.clip_seconds if args.clip_seconds and args.clip_seconds > 0 else None

    run_pipeline(
        source_root=args.source_root,
        target_root=args.target_root,
        output_dir=args.output_dir,
        include_source=not args.skip_source,
        include_target=not args.skip_target,
        include_sensors=args.sensors,
        include_datasets=args.source_datasets,
        target_sample_rate=target_rate,
        clip_seconds=clip_seconds,
    )


if __name__ == "__main__":
    main()
