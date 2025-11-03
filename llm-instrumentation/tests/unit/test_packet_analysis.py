import struct

import matplotlib
import pandas as pd
import torch
import matplotlib.pyplot as plt

from llm_instrumentation.analysis import (
    analyze_circuits,
    build_packet_dataframe,
    create_interactive_dashboard,
    load_stream_packets,
    plot_token_progression,
    visualize_attention_patterns,
)
from llm_instrumentation.core.compression import TensorCompressionManager

matplotlib.use("Agg")


def _write_sample_stream(path: str, compression: str = "lz4", num_packets: int = 6) -> None:
    """Create a small synthetic stream file for analysis tests."""
    manager = TensorCompressionManager()
    manager.current_strategy = compression

    with open(path, "wb") as f:
        for idx in range(num_packets):
            tensor = torch.linspace(0, 1, steps=64, dtype=torch.float32).reshape(8, 8)
            tensor = tensor + idx * 0.1  # ensure variance across packets
            compressed, _ = manager.compress_tensor(tensor)
            layer_name = f"layer_{idx % 3}"

            header = struct.pack("!HI", len(layer_name), len(compressed))
            f.write(header)
            f.write(layer_name.encode("utf-8"))
            f.write(compressed)


def test_packet_analysis_pipeline(tmp_path):
    stream_path = tmp_path / "synthetic.stream"
    _write_sample_stream(str(stream_path))

    packets = load_stream_packets(str(stream_path), compression="lz4")
    assert len(packets) > 0
    assert {"layer_name", "compressed_bytes", "l2_norm"} <= packets[0].keys()

    token_metadata = {
        "total_tokens": 3,
        "tokens": [{"token_text": f"tok-{i}"} for i in range(3)],
    }

    df = build_packet_dataframe(packets, token_metadata=token_metadata)
    assert isinstance(df, pd.DataFrame)
    assert "token_index" in df.columns
    assert df["token_index"].max() <= token_metadata["total_tokens"] - 1

    ax = plot_token_progression(df, token_metadata=token_metadata)
    assert ax is not None

    ax_heatmap = visualize_attention_patterns(df, token_metadata=token_metadata)
    assert ax_heatmap is not None

    circuits = analyze_circuits(df)
    assert isinstance(circuits, pd.DataFrame)
    assert set(circuits.columns) == {"layer_a", "layer_b", "correlation"}

    dashboard = create_interactive_dashboard(df)
    assert dashboard is not None

    plt.close("all")
