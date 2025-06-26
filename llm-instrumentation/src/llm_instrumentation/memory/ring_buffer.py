import torch
import threading


class RingBuffer:
    """
    A thread-safe ring buffer for byte data using a PyTorch tensor.
    """

    def __init__(self, size: int):
        """
        Initializes the RingBuffer.

        Args:
            size (int): The size of the buffer in bytes.
        """
        if size <= 0:
            raise ValueError("Buffer size must be positive")
        self.size = size
        self.buffer = torch.empty(size, dtype=torch.uint8)
        self.write_pos = 0
        self.read_pos = 0
        self.bytes_available = 0
        self.lock = threading.Lock()
        self.data_available = threading.Condition(self.lock)
        self.space_available = threading.Condition(self.lock)

    def put(self, data: bytes, timeout: float = 0.1) -> bool:
        """
        Puts data into the buffer. Blocks if the buffer is full.

        Args:
            data (bytes): The data to put into the buffer.
            timeout (float): The maximum time to wait in seconds.

        Returns:
            bool: True if the data was put into the buffer, False if it
                  timed out.
        """
        data_len = len(data)
        if data_len > self.size:
            raise ValueError("Data size exceeds buffer capacity")

        tensor_data = torch.frombuffer(bytearray(data), dtype=torch.uint8)

        with self.space_available:
            if self.size - self.bytes_available < data_len:
                if not self.space_available.wait_for(
                    lambda: self.size - self.bytes_available >= data_len,
                    timeout=timeout,
                ):
                    return False

            # Write data, potentially in two parts if it wraps around
            remaining_space_at_end = self.size - self.write_pos
            if data_len <= remaining_space_at_end:
                write_slice = slice(self.write_pos, self.write_pos + data_len)
                self.buffer[write_slice] = tensor_data
            else:
                part1_len = remaining_space_at_end
                write_slice1 = slice(self.write_pos, self.write_pos + part1_len)
                self.buffer[write_slice1] = tensor_data[:part1_len]
                part2_len = data_len - part1_len
                self.buffer[0:part2_len] = tensor_data[part1_len:]

            self.write_pos = (self.write_pos + data_len) % self.size
            self.bytes_available += data_len
            self.data_available.notify()
            return True

    def get(self, max_bytes: int, timeout: float = 0.1) -> bytes:
        """
        Gets data from the buffer. Blocks if the buffer is empty.

        Args:
            max_bytes (int): The maximum number of bytes to read.
            timeout (float): The maximum time to wait in seconds.

        Returns:
            bytes: The data read from the buffer. Returns empty bytes if it
                   timed out.
        """
        with self.data_available:
            if self.bytes_available == 0:
                if not self.data_available.wait(timeout=timeout):
                    return b""

            bytes_to_read = min(max_bytes, self.bytes_available)

            # Read data, potentially in two parts if it wraps around
            remaining_data_at_end = self.size - self.read_pos
            if bytes_to_read <= remaining_data_at_end:
                read_slice = slice(self.read_pos, self.read_pos + bytes_to_read)
                data_tensor = self.buffer[read_slice]
            else:
                part1_len = remaining_data_at_end
                read_slice1 = slice(self.read_pos, self.read_pos + part1_len)
                part1 = self.buffer[read_slice1]
                part2_len = bytes_to_read - part1_len
                part2 = self.buffer[0:part2_len]
                data_tensor = torch.cat((part1, part2))

            self.read_pos = (self.read_pos + bytes_to_read) % self.size
            self.bytes_available -= bytes_to_read
            self.space_available.notify()

            return data_tensor.numpy().tobytes()

    def __len__(self) -> int:
        """Returns the number of bytes currently in the buffer."""
        with self.lock:
            return self.bytes_available

    @property
    def capacity(self) -> int:
        """Returns the total capacity of the buffer."""
        return self.size

    @property
    def free_space(self) -> int:
        """Returns the number of free bytes in the buffer."""
        with self.lock:
            return self.size - self.bytes_available
