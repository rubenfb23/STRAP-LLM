import asyncio
import struct
import pytest
import torch
from unittest.mock import MagicMock

from llm_instrumentation.core.streaming import StreamingSerializer


@pytest.fixture
def serializer():
    """Fixture for a StreamingSerializer instance."""
    return StreamingSerializer(buffer_size=1024, num_workers=1)


@pytest.mark.asyncio
async def test_streaming_start_stop(serializer: StreamingSerializer):
    """Tests the start and stop methods of the serializer."""
    assert not serializer.running
    await serializer.start_streaming("test_output.bin")
    assert serializer.running
    assert serializer.writer_task is not None
    assert len(serializer.compression_tasks) == 1

    await serializer.stop_streaming()
    assert not serializer.running
    # Ensure tasks are cleaned up
    assert serializer.writer_task.done()
    for task in serializer.compression_tasks:
        assert task.done()


@pytest.mark.asyncio
async def test_enqueue_tensor_and_write(serializer: StreamingSerializer, tmp_path):
    """Tests that a tensor is enqueued, compressed, and written to a file."""
    output_file = tmp_path / "test_output.bin"

    # Mock the compressor to return predictable data
    mock_compressor = MagicMock()
    compressed_data = b"compressed_tensor_data"
    mock_compressor.compress_tensor.return_value = (compressed_data, 2.0)
    serializer.compressor = mock_compressor

    await serializer.start_streaming(str(output_file))

    layer_name = "test_layer"
    tensor = torch.randn(4, 4)

    await serializer.enqueue_tensor(layer_name, tensor)

    await serializer.stop_streaming()

    # Verify the file content
    assert output_file.exists()
    with open(output_file, "rb") as f:
        content = f.read()

    # Check header and data
    layer_name_bytes = layer_name.encode("utf-8")
    expected_header = struct.pack("!HI", len(layer_name_bytes), len(compressed_data))
    expected_packet = expected_header + layer_name_bytes + compressed_data

    assert content == expected_packet


@pytest.mark.asyncio
async def test_multiple_tensors(serializer: StreamingSerializer, tmp_path):
    """Tests streaming multiple tensors."""
    output_file = tmp_path / "test_output.bin"
    await serializer.start_streaming(str(output_file))

    tensors = {
        "layer1": torch.ones(2, 2),
        "layer2": torch.zeros(3, 3),
        "layer3": torch.full((4, 4), 5.0),
    }

    for name, tensor in tensors.items():
        await serializer.enqueue_tensor(name, tensor)

    await serializer.stop_streaming()

    assert output_file.exists()
    with open(output_file, "rb") as f:
        for name, tensor in tensors.items():
            # Read header
            header_data = f.read(struct.calcsize("!HI"))
            name_len, data_len = struct.unpack("!HI", header_data)

            # Read layer name
            read_name = f.read(name_len).decode("utf-8")
            assert read_name == name

            # Read data
            data = f.read(data_len)
            assert len(data) == data_len


@pytest.mark.asyncio
async def test_stop_streaming_waits_for_queue(tmp_path):
    """Tests that stop_streaming waits for the queue to be empty."""
    output_file = tmp_path / "test_output.bin"
    # Use a real serializer but with a mocked compression worker to add delays
    serializer = StreamingSerializer(buffer_size=10240, num_workers=1)

    original_worker = serializer._compression_worker

    async def delayed_worker():
        # Simulate work
        await asyncio.sleep(0.2)
        await original_worker()

    # We can't easily mock the worker directly, so we'll test the outcome:
    # all items enqueued should be written to the file.

    await serializer.start_streaming(str(output_file))

    num_tensors = 5
    for i in range(num_tensors):
        await serializer.enqueue_tensor(f"layer_{i}", torch.randn(2, 2))

    await serializer.stop_streaming()

    assert output_file.exists()
    with open(output_file, "rb") as f:
        num_packets = 0
        while True:
            header_data = f.read(struct.calcsize("!HI"))
            if not header_data:
                break
            name_len, data_len = struct.unpack("!HI", header_data)
            f.read(name_len)  # Skip name
            f.read(data_len)  # Skip data
            num_packets += 1

    assert num_packets == num_tensors
    # A better assertion:
    assert serializer.input_queue.empty()
