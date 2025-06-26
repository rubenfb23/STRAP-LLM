import pytest
import threading
import time
from llm_instrumentation.memory.ring_buffer import RingBuffer


def test_ring_buffer_initialization():
    """Tests RingBuffer initialization."""
    buffer = RingBuffer(1024)
    assert buffer.capacity == 1024
    assert len(buffer) == 0
    assert buffer.free_space == 1024

    with pytest.raises(ValueError):
        RingBuffer(0)

    with pytest.raises(ValueError):
        RingBuffer(-1)


def test_ring_buffer_simple_put_get():
    """Tests simple put and get operations."""
    buffer = RingBuffer(1024)
    data_in = b"hello world"
    assert buffer.put(data_in) is True
    assert len(buffer) == len(data_in)
    assert buffer.free_space == 1024 - len(data_in)

    data_out = buffer.get(len(data_in))
    assert data_out == data_in
    assert len(buffer) == 0
    assert buffer.free_space == 1024


def test_ring_buffer_wrap_around():
    """Tests that the buffer correctly wraps around."""
    buffer = RingBuffer(16)
    data1 = b"12345678"
    data2 = b"abcdefgh"
    data3 = b"ijklmnop"

    # Fill the buffer to the end
    buffer.put(data1)  # write_pos = 8
    buffer.put(data2)  # write_pos = 0 (wrap)

    # Read first part, making space at the beginning
    assert buffer.get(8) == data1

    # This write should wrap around
    assert buffer.put(data3) is True

    assert buffer.get(8) == data2
    assert buffer.get(8) == data3


def test_ring_buffer_put_timeout():
    """Tests that put times out correctly when the buffer is full."""
    buffer = RingBuffer(10)
    assert buffer.put(b"1234567890") is True

    start_time = time.time()
    assert buffer.put(b"a", timeout=0.1) is False
    end_time = time.time()
    assert end_time - start_time >= 0.1


def test_ring_buffer_get_timeout():
    """Tests that get times out correctly when the buffer is empty."""
    buffer = RingBuffer(10)

    start_time = time.time()
    assert buffer.get(1, timeout=0.1) == b""
    end_time = time.time()
    assert end_time - start_time >= 0.1


def test_ring_buffer_put_too_large():
    """Tests that putting data larger than the buffer raises an error."""
    buffer = RingBuffer(8)
    with pytest.raises(ValueError):
        buffer.put(b"123456789")


def test_ring_buffer_threaded_access():
    """Tests thread-safe access to the ring buffer."""
    buffer = RingBuffer(1024 * 1024)  # 1MB buffer
    data_chunk = b"x" * 1024  # 1KB chunks
    num_chunks = 500
    total_data_size = len(data_chunk) * num_chunks

    producer_finished = threading.Event()

    def producer():
        for _ in range(num_chunks):
            assert buffer.put(data_chunk, timeout=1)
        producer_finished.set()

    def consumer():
        bytes_read = 0
        while not (producer_finished.is_set() and len(buffer) == 0):
            data = buffer.get(len(data_chunk), timeout=0.1)
            if data:
                bytes_read += len(data)
        return bytes_read

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    # The test fails if the consumer is not started before the producer
    # because the producer can finish before the consumer starts, leading
    # to a race condition where the test finishes before the consumer
    # has read any data.
    # To fix this, we need to ensure the consumer is ready to read data
    # before the producer starts writing.
    consumer_ready = threading.Event()

    def consumer_with_ready_signal():
        bytes_read = 0
        consumer_ready.set()
        while not (producer_finished.is_set() and len(buffer) == 0):
            data = buffer.get(len(data_chunk), timeout=0.1)
            if data:
                bytes_read += len(data)
        return bytes_read

    consumer_thread = threading.Thread(target=consumer_with_ready_signal)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    # This assertion is problematic because the consumer_thread returns the
    # number of bytes read, but we are not capturing that return value.
    # We need to modify the test to capture the return value from the
    # consumer thread.
    # A simple way is to use a list to store the result.
    result = []

    def consumer_with_result(res_list):
        bytes_read = 0
        consumer_ready.set()
        while not (producer_finished.is_set() and len(buffer) == 0):
            data = buffer.get(len(data_chunk), timeout=0.1)
            if data:
                bytes_read += len(data)
        res_list.append(bytes_read)

    producer_finished.clear()
    consumer_ready.clear()

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer_with_result, args=(result,))

    consumer_thread.start()
    consumer_ready.wait()  # Ensure consumer is ready
    producer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    assert result[0] == total_data_size
