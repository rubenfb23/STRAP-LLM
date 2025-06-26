# STRAP-LLM: Streaming Transformer Activation Pipeline

STRAP-LLM is a high-performance framework for capturing and analyzing the internal activations of large language models (LLMs). It is designed for researchers and developers who need to understand the inner workings of these complex models for interpretability, debugging, or optimization purposes.

This repository contains the core `llm-instrumentation` library, which provides the tools for instrumenting PyTorch-based transformer models.

## Key Features

- **High-Throughput Streaming**: Captures model activations with minimal impact on inference speed.
- **Efficient Data Handling**: Utilizes asynchronous I/O and compression to manage large volumes of activation data.
- **Extensible Analysis Tools**: Provides a foundation for building sophisticated interpretability analyses, such as causal tracing and feature extraction with sparse autoencoders.
- **Modular Design**: Easily configurable to target specific layers and model types.

## Getting Started

The core functionality is contained within the `llm-instrumentation` directory. For detailed information on how to install, use, and understand the framework, please refer to the documentation within that directory.

- **[Project Details and Documentation](./llm-instrumentation/README.md)**

## Repository Structure

- **`llm-instrumentation/`**: The main Python package containing the instrumentation and analysis framework.
  - **`src/`**: The source code for the framework.
  - **`docs/`**: Detailed documentation on the architecture, API, and performance.
  - **`examples/`**: Sample scripts demonstrating how to use the framework.
  - **`benchmarks/`**: Performance testing scripts.
- **`LICENSE`**: The project license.
- **`README.md`**: This file.

## Contributing

Contributions are welcome! Please see the `llm-instrumentation/README.md` for more information on the development setup and contribution process.
