# System Architecture

This document provides a detailed overview of the system architecture for the LLM Instrumentation Framework.

The LLM Instrumentation framework is designed for high-performance capture and analysis of transformer model activations. The architecture is modular, allowing for flexible configuration and extension.

## Core Components

The framework consists of several core components:

- **Instrumentation Core (`llm_instrumentation.core`)**: This is the heart of the framework, responsible for hooking into the model and capturing activations.
    - `hooks.py`: Manages the PyTorch hooks that capture the outputs of specified model layers.
    - `streaming.py`: Handles the high-throughput streaming of captured activations to a persistent storage medium.
    - `compression.py`: Provides various compression algorithms (e.g., LZ4, Zstd) to reduce the volume of data being streamed.
    - `profiling.py`: Contains tools for profiling the performance of the instrumentation framework itself.

- **Memory Management (`llm_instrumentation.memory`)**: Efficient memory management is crucial for handling the large volume of activation data.
    - `ring_buffer.py`: A CUDA-based ring buffer for temporarily storing activations in GPU memory before they are processed and streamed.
    - `async_writer.py`: An asynchronous writer that moves data from the GPU to host memory and then to disk, minimizing I/O latency.
    - `cuda_manager.py`: A utility for managing CUDA memory allocations.

- **Analysis Engine (`llm_instrumentation.analysis`)**: This component provides tools for offline analysis of the captured activation data.
    - `causal_graph.py`: Tools for constructing and analyzing causal graphs from the activation data to understand information flow within the model.
    - `sparse_autoencoder.py`: Functionality for training and applying sparse autoencoders to find interpretable features in the activations.
    - `visualization.py`: Tools for visualizing the results of the analysis.

- **Utilities (`llm_instrumentation.utils`)**:
    - `config.py`: Manages configuration for the instrumentation and analysis.
    - `logging.py`: Provides logging facilities.
    - `metrics.py`: For collecting and reporting performance metrics.

## Data Flow

1.  The `InstrumentationFramework` uses `hooks.py` to attach forward hooks to the specified layers of the transformer model.
2.  During the model's forward pass, the hooks capture the activation tensors.
3.  These tensors are placed into the `CudaRingBuffer`.
4.  A background worker thread retrieves tensors from the ring buffer, compresses them using one of the configured `compression` algorithms, and moves them to a host memory buffer.
5.  The `AsyncWriter` then writes the compressed data from the host buffer to disk.
6.  This entire process is designed to be asynchronous to minimize the impact on the model's inference speed.
7.  Once the data is captured, the `analysis` modules can be used to load the data and perform various interpretability studies.
