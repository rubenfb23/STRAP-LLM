import json
import os

import pytest

from llm_instrumentation import (
    HookGranularity,
    InstrumentationConfig,
    InstrumentationFramework,
)


def test_token_tracking_disabled_by_default(tmp_path):
    """Test that token tracking is opt-in."""
    config = InstrumentationConfig(
        granularity=HookGranularity.FULL_TENSOR,
        compression_algorithm="none",
    )
    framework = InstrumentationFramework(config)

    output_path = tmp_path / "test.stream"

    with framework.capture_activations(str(output_path)) as tracker:
        assert tracker is None

    metadata_path = str(output_path).replace(".stream", "_tokens.json")
    assert not os.path.exists(metadata_path)


def test_token_tracking_enabled(tmp_path):
    """Test that token tracking creates metadata file."""
    config = InstrumentationConfig(
        granularity=HookGranularity.FULL_TENSOR,
        compression_algorithm="none",
    )
    framework = InstrumentationFramework(config)

    output_path = tmp_path / "test.stream"

    with framework.capture_activations(str(output_path), track_per_token=True) as tracker:
        assert tracker is not None

        tracker.record_token(123, "hello")
        tracker.record_token(456, "world")

    metadata_path = str(output_path).replace(".stream", "_tokens.json")
    assert os.path.exists(metadata_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["total_tokens"] == 2
    assert len(metadata["tokens"]) == 2
    assert metadata["tokens"][0]["token_id"] == 123
    assert metadata["tokens"][0]["token_text"] == "hello"
    assert metadata["tokens"][1]["token_id"] == 456


def test_tracker_auto_position(tmp_path):
    """Test that token positions are auto-computed."""
    config = InstrumentationConfig(
        granularity=HookGranularity.FULL_TENSOR,
        compression_algorithm="none",
    )
    framework = InstrumentationFramework(config)

    output_path = tmp_path / "test.stream"

    with framework.capture_activations(str(output_path), track_per_token=True) as tracker:
        tracker.record_token(1, "a")
        tracker.record_token(2, "b")
        tracker.record_token(3, "c")

    metadata_path = str(output_path).replace(".stream", "_tokens.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["tokens"][0]["token_index"] == 0
    assert metadata["tokens"][1]["token_index"] == 1
    assert metadata["tokens"][2]["token_index"] == 2
