# Stream Format

This document describes the on-disk format written by the `StreamingSerializer`.

Each activation tensor is serialized as a self-contained packet with a fixed header, UTF-8 layer name, and compressed data payload.

Packet layout (Big Endian / network order):

- Header: `!HI`
  - `name_len` (`uint16`): Length of the layer name in bytes.
  - `data_len` (`uint32`): Length of the compressed payload in bytes.
- Name: `name_len` bytes, UTF-8 encoded layer/module name.
- Data: `data_len` bytes of compressed tensor payload.

Notes

- Packets are back-to-back with no additional framing.
- Compression algorithm is configured at runtime. The current algorithm is not embedded per-packet; keep run metadata alongside the stream.
- Tensors are downcast to FP16 when source dtype is FP32 prior to compression.

Minimal parser example

```python
import struct

def iter_packets(path):
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(struct.calcsize('!HI'))
            if not hdr:
                break
            name_len, data_len = struct.unpack('!HI', hdr)
            name = f.read(name_len).decode('utf-8')
            data = f.read(data_len)
            yield name, data

for name, payload in iter_packets('output.stream'):
    print(name, len(payload))
```

Decompression

- LZ4: `lz4.frame.decompress(payload)`
- Zstd: `zstandard.ZstdDecompressor().decompress(payload)`
- None: Payload is raw serialized FP16 bytes of the tensor.

Tensor reloading is application-specific. The serializer stores the raw tensor buffer; shape and dtype metadata are not embedded in the packet. If you need full fidelity, log shape/dtype metadata out-of-band (e.g., alongside the stream filename) or encode it into the layer name by convention.

