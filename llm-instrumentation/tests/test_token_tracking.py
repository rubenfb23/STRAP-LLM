import json
import os
import time

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


def test_checkpoint_interval_creates_persistent_snapshot(tmp_path):
    """Token checkpoints should be flushed once the interval is reached."""
    config = InstrumentationConfig(
        granularity=HookGranularity.FULL_TENSOR,
        compression_algorithm="none",
    )
    framework = InstrumentationFramework(config)

    output_path = tmp_path / "checkpoint.stream"
    checkpoint_path = tmp_path / "checkpoint.stream.ckpt.json"

    with framework.capture_activations(
        str(output_path),
        track_per_token=True,
        checkpoint_interval_tokens=2,
    ) as tracker:
        tracker.record_token(10, "alpha")
        assert not checkpoint_path.exists()

        tracker.record_token(11, "beta")
        assert checkpoint_path.exists()

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)

        assert checkpoint_data["total_tokens"] == 2
        assert len(checkpoint_data["tokens"]) == 2
        assert checkpoint_data["tokens"][1]["token_text"] == "beta"

    # Checkpoint should be cleaned up after successful completion
    assert not checkpoint_path.exists()


def test_resume_from_checkpoint_restores_previous_tokens(tmp_path):
    """Resume mode should hydrate tokens from an existing checkpoint."""
    config = InstrumentationConfig(
        granularity=HookGranularity.FULL_TENSOR,
        compression_algorithm="none",
    )
    framework = InstrumentationFramework(config)

    output_path = tmp_path / "resume.stream"
    checkpoint_path = tmp_path / "resume.stream.ckpt.json"

    initial_tokens = [
        {"token_index": 0, "token_id": 100, "token_text": "tok0"},
        {"token_index": 1, "token_id": 101, "token_text": "tok1"},
    ]
    start_time = time.time() - 5
    checkpoint_payload = {
        "version": 1,
        "total_tokens": len(initial_tokens),
        "last_checkpoint_time": start_time,
        "metadata": {
            "start_time": start_time,
            "resume_count": 0,
            "updated_at": start_time,
        },
        "tokens": initial_tokens,
    }

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_payload, f)

    with framework.capture_activations(
        str(output_path),
        track_per_token=True,
        checkpoint_interval_tokens=1,
        resume_from_checkpoint=True,
    ) as tracker:
        assert tracker is not None
        assert len(tracker.tokens) == 2  # hydrated from checkpoint

        tracker.record_token(102, "tok2")
        assert checkpoint_path.exists()

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            updated_ckpt = json.load(f)

        assert updated_ckpt["total_tokens"] == 3
        assert updated_ckpt["metadata"]["resume_count"] == 1

        tracker.record_token(103, "tok3")

    metadata_path = str(output_path).replace(".stream", "_tokens.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["total_tokens"] == 4
    assert metadata["tokens"][0]["token_id"] == 100
    assert metadata["tokens"][3]["token_text"] == "tok3"
    assert not checkpoint_path.exists()


def test_checkpoint_requires_token_tracking(tmp_path):
    """Checkpoint configuration without token tracking should raise."""
    config = InstrumentationConfig(
        granularity=HookGranularity.FULL_TENSOR,
        compression_algorithm="none",
    )
    framework = InstrumentationFramework(config)

    output_path = tmp_path / "invalid.stream"

    with pytest.raises(ValueError):
        with framework.capture_activations(
            str(output_path),
            track_per_token=False,
            checkpoint_interval_tokens=1,
        ):
            pass

    with pytest.raises(ValueError):
        with framework.capture_activations(
            str(output_path),
            track_per_token=False,
            resume_from_checkpoint=True,
        ):
            pass
