"""High-level helpers for interpreting activation stream captures.

These utilities operate on `.stream` files emitted by the instrumentation
framework and provide ergonomic ways to inspect per-layer statistics, correlate
activations with generated tokens, and visualise trends for interpretability
workflows used in the notebooks.
"""

from __future__ import annotations

import math
import struct
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from llm_instrumentation.core.compression import TensorCompressionManager

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised only when pandas missing
    pd = None  # type: ignore


PacketLike = Dict[str, Any]


def load_stream_packets(
    stream_path: str,
    compression: str,
    include_failed: bool = False,
) -> List[PacketLike]:
    """Parse a `.stream` file into a list of packet records.

    Args:
        stream_path: Path to the activation stream file.
        compression: Compression algorithm used during capture.
        include_failed: If True, include packets that could not be decompressed.

    Returns:
        A list with one dictionary per packet containing layer metadata and
        summary statistics (L2 norm, mean, std, etc.).
    """
    manager = TensorCompressionManager()
    manager.current_strategy = compression

    packet_records: List[PacketLike] = []
    header_fmt = "!HI"
    header_size = struct.calcsize(header_fmt)

    with open(stream_path, "rb") as f:
        packet_index = 0
        while True:
            header = f.read(header_size)
            if not header:
                break
            if len(header) != header_size:
                warnings.warn("Encountered truncated packet header; stopping parse.")
                break

            name_len, data_len = struct.unpack(header_fmt, header)
            layer_name_bytes = f.read(name_len)
            if len(layer_name_bytes) != name_len:
                warnings.warn("Encountered truncated layer name; stopping parse.")
                break

            compressed_payload = f.read(data_len)
            if len(compressed_payload) != data_len:
                warnings.warn("Encountered truncated payload; stopping parse.")
                break

            layer_name = layer_name_bytes.decode("utf-8") or "<root>"
            packet_info: PacketLike = {
                "packet_index": packet_index,
                "layer_name": layer_name,
                "compressed_bytes": data_len,
            }

            try:
                decompressed = manager.decompress_tensor(compressed_payload, strategy=compression)
                packet_info["uncompressed_bytes"] = len(decompressed)
                if len(decompressed) % 2 != 0:
                    # Align to even length for float16 interpretation
                    decompressed = decompressed[:-1]
                if decompressed:
                    values = np.frombuffer(decompressed, dtype=np.float16).astype(np.float32)
                    packet_info.update(_compute_tensor_statistics(values))
                else:
                    packet_info.update(_empty_tensor_statistics())
            except Exception as exc:  # pragma: no cover - depends on runtime tensor mix
                packet_info.update(_empty_tensor_statistics())
                packet_info["error"] = str(exc)
                if not include_failed:
                    packet_index += 1
                    continue

            packet_records.append(packet_info)
            packet_index += 1

    return packet_records


def build_packet_dataframe(
    stream_packets: Union[str, Sequence[PacketLike]],
    token_metadata: Optional[Dict[str, Any]] = None,
) -> "pd.DataFrame":
    """Convert parsed packets into a pandas DataFrame with optional token mapping."""

    if pd is None:  # pragma: no cover - pandas absent only in constrained envs
        raise ImportError(
            "pandas is required for build_packet_dataframe(). "
            "Install the 'analysis' extra: pip install llm-instrumentation[analysis]"
        )

    if isinstance(stream_packets, str):
        raise ValueError(
            "build_packet_dataframe expects pre-parsed packet records. "
            "Use load_stream_packets(path, compression) first."
        )

    if not stream_packets:
        return pd.DataFrame(
            columns=[
                "packet_index",
                "layer_name",
                "compressed_bytes",
                "uncompressed_bytes",
                "compression_ratio",
                "l2_norm",
                "mean",
                "std",
                "max",
                "min",
                "sparsity",
            ]
        )

    df = pd.DataFrame(stream_packets)
    if "compression_ratio" not in df.columns and "uncompressed_bytes" in df.columns:
        df["compression_ratio"] = np.where(
            df["compressed_bytes"] > 0,
            df["uncompressed_bytes"] / df["compressed_bytes"],
            np.nan,
        )

    if token_metadata:
        df = _assign_tokens(df, token_metadata)

    return df.sort_values("packet_index").reset_index(drop=True)


def plot_token_progression(
    packet_stats: "pd.DataFrame",
    token_metadata: Optional[Dict[str, Any]] = None,
    metric: str = "l2_norm",
    ax=None,
):
    """Plot how a metric evolves across generated tokens."""

    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for plotting utilities.")

    if metric not in packet_stats.columns:
        raise ValueError(f"Metric '{metric}' not present in packet statistics.")

    if "token_index" not in packet_stats.columns and token_metadata:
        packet_stats = _assign_tokens(packet_stats, token_metadata)

    if "token_index" not in packet_stats.columns:
        raise ValueError(
            "Token indices are unavailable. Provide token_metadata from "
            "analyze_activations_with_tokens to enable progression plots."
        )

    grouped = (
        packet_stats.groupby("token_index")[metric]
        .mean()
        .reset_index()
        .sort_values("token_index")
    )

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(grouped["token_index"], grouped[metric], marker="o", linewidth=1.5)
    ax.set_xlabel("Token Index")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} Progression per Token")
    ax.grid(True, linestyle="--", alpha=0.3)

    if token_metadata and "tokens" in token_metadata:
        xticks = grouped["token_index"].tolist()
        labels = []
        tokens = token_metadata.get("tokens", [])
        for idx in xticks:
            if 0 <= idx < len(tokens):
                text = tokens[idx].get("token_text", "").strip()
                labels.append(text or f"#{idx}")
            else:
                labels.append(f"#{idx}")
        if len(xticks) <= 20:
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels, rotation=45, ha="right")

    return ax


def visualize_attention_patterns(
    stream_packets: Union["pd.DataFrame", Sequence[PacketLike]],
    token_metadata: Optional[Dict[str, Any]] = None,
    metric: str = "l2_norm",
    ax=None,
):
    """Render a heatmap of activation intensity per layer/token."""

    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for visualization utilities.")

    if not isinstance(stream_packets, pd.DataFrame):
        packet_stats = build_packet_dataframe(stream_packets, token_metadata)
    else:
        packet_stats = stream_packets.copy()
        if "token_index" not in packet_stats.columns and token_metadata:
            packet_stats = _assign_tokens(packet_stats, token_metadata)

    if "token_index" not in packet_stats.columns:
        raise ValueError(
            "Token indices are unavailable. Provide token_metadata from "
            "analyze_activations_with_tokens to enable heatmap visualization."
        )

    pivot = (
        packet_stats.pivot_table(
            index="token_index",
            columns="layer_name",
            values=metric,
            aggfunc="mean",
        )
        .fillna(0.0)
        .sort_index()
    )

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Token Index")
    ax.set_title(f"{metric.replace('_', ' ').title()} Heatmap (Token × Layer)")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=metric.replace("_", " ").title())
    return ax


def analyze_circuits(
    packet_stats: "pd.DataFrame",
    metric: str = "l2_norm",
    min_correlation: float = 0.6,
    top_k: int = 8,
) -> "pd.DataFrame":
    """Identify pairs of layers with highly correlated activation trajectories."""

    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for analyze_circuits.")

    if metric not in packet_stats.columns:
        raise ValueError(f"Metric '{metric}' not present in packet statistics.")

    if "token_index" not in packet_stats.columns:
        raise ValueError("Token alignment required. Use packet_stats with token_index.")

    pivot = (
        packet_stats.pivot_table(
            index="token_index",
            columns="layer_name",
            values=metric,
            aggfunc="mean",
        )
        .dropna(axis=1, how="all")
    )
    if pivot.empty or pivot.shape[1] < 2:
        return pd.DataFrame(columns=["layer_a", "layer_b", "correlation"])

    corr = pivot.corr()
    records: List[Tuple[str, str, float]] = []
    for i, col_a in enumerate(corr.columns):
        for j, col_b in enumerate(corr.columns):
            if j <= i:
                continue
            value = corr.iloc[i, j]
            if math.isnan(value) or abs(value) < min_correlation:
                continue
            records.append((col_a, col_b, float(value)))

    results = pd.DataFrame(records, columns=["layer_a", "layer_b", "correlation"])
    if results.empty:
        return results
    return results.sort_values("correlation", ascending=False).head(top_k).reset_index(drop=True)


def create_interactive_dashboard(
    packet_stats: "pd.DataFrame",
    metric: str = "l2_norm",
    secondary_metric: str = "compression_ratio",
):
    """Create an interactive (Plotly if available, matplotlib otherwise) dashboard."""

    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for dashboard utilities.")

    if metric not in packet_stats.columns:
        raise ValueError(f"Metric '{metric}' not present in packet statistics.")

    if "token_index" not in packet_stats.columns:
        warnings.warn(
            "Token indices missing; dashboard will aggregate over packet order only."
        )

    try:  # Prefer Plotly for interactivity
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - fallback path tested separately
        return _matplotlib_dashboard(packet_stats, metric, secondary_metric)

    df = packet_stats.copy()
    df["packet_order"] = np.arange(len(df))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[
            f"{metric.replace('_', ' ').title()} per Packet",
            f"{secondary_metric.replace('_', ' ').title()} by Layer",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=df["packet_order"],
            y=df[metric],
            mode="lines+markers",
            hovertext=df["layer_name"],
            name=metric,
        ),
        row=1,
        col=1,
    )

    layer_summary = (
        df.groupby("layer_name")[secondary_metric]
        .mean()
        .reset_index()
        .sort_values(secondary_metric, ascending=False)
    )
    fig.add_trace(
        go.Bar(
            x=layer_summary["layer_name"],
            y=layer_summary[secondary_metric],
            name=secondary_metric,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(title_text="Packet Order", row=1, col=1)
    fig.update_yaxes(title_text=metric.replace("_", " ").title(), row=1, col=1)
    fig.update_yaxes(title_text=secondary_metric.replace("_", " ").title(), row=2, col=1)
    fig.update_xaxes(title_text="Layer", row=2, col=1)
    return fig


def _matplotlib_dashboard(packet_stats: "pd.DataFrame", metric: str, secondary_metric: str):
    """Fallback dashboard using matplotlib (non-interactive but informative)."""
    import matplotlib.pyplot as plt

    df = packet_stats.copy()
    df["packet_order"] = np.arange(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    axes[0].plot(df["packet_order"], df[metric], linewidth=1.5)
    axes[0].set_title(f"{metric.replace('_', ' ').title()} per Packet")
    axes[0].set_xlabel("Packet Order")
    axes[0].set_ylabel(metric.replace("_", " ").title())
    axes[0].grid(True, linestyle="--", alpha=0.3)

    layer_summary = (
        df.groupby("layer_name")[secondary_metric]
        .mean()
        .reset_index()
        .sort_values(secondary_metric, ascending=False)
    )
    axes[1].bar(layer_summary["layer_name"], layer_summary[secondary_metric])
    axes[1].set_title(f"{secondary_metric.replace('_', ' ').title()} by Layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel(secondary_metric.replace("_", " ").title())
    axes[1].tick_params(axis="x", rotation=90)

    plt.tight_layout()
    warnings.warn(
        "Plotly not installed; falling back to matplotlib dashboard (static figure)."
    )
    return fig


def _compute_tensor_statistics(values: np.ndarray) -> PacketLike:
    """Compute scalar statistics for a tensor represented as a flat array."""
    if values.size == 0:
        return _empty_tensor_statistics()
    squared_norm = float(np.linalg.norm(values))
    return {
        "l2_norm": squared_norm,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "sparsity": float(np.mean(values == 0.0)),
    }


def _empty_tensor_statistics() -> PacketLike:
    """Return a statistics placeholder for empty tensors."""
    return {
        "l2_norm": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "max": np.nan,
        "min": np.nan,
        "sparsity": np.nan,
    }


def _assign_tokens(packet_df: "pd.DataFrame", token_metadata: Dict[str, Any]) -> "pd.DataFrame":
    """Assign token indices/text to packets using proportional distribution."""
    total_tokens = int(token_metadata.get("total_tokens") or 0)
    if total_tokens <= 0:
        return packet_df

    num_packets = len(packet_df)
    base = num_packets // total_tokens
    remainder = num_packets % total_tokens

    token_indices: List[int] = []
    for idx in range(total_tokens):
        count = base + (1 if idx < remainder else 0)
        token_indices.extend([idx] * count)
    if len(token_indices) < num_packets:
        token_indices.extend([total_tokens - 1] * (num_packets - len(token_indices)))

    packet_df = packet_df.copy()
    packet_df["token_index"] = token_indices[:num_packets]

    tokens = token_metadata.get("tokens") or []
    if tokens:
        def _token_text(i: int) -> Optional[str]:
            if 0 <= i < len(tokens):
                return tokens[i].get("token_text")
            return None

        packet_df["token_text"] = packet_df["token_index"].apply(_token_text)

    return packet_df
