import torch
import torch.nn as nn
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from llm_instrumentation.core.hooks import (
    HookConfig,
    HookGranularity,
    OptimizedHookManager,
    TensorProcessor,
)


class MockTensorProcessor(TensorProcessor):
    def __init__(self):
        self.processed_data = []

    def process(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> bytes:
        self.processed_data.append((tensor, metadata))
        return b"processed"


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Sequential(nn.Linear(10, 10), nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def test_attach_and_detach_hooks():
    model = SimpleModel()
    tensor_processor = MockTensorProcessor()
    manager = OptimizedHookManager(tensor_processor)
    config = HookConfig(granularity=HookGranularity.FULL_TENSOR, async_enabled=False)

    manager.attach_hooks(model, config)
    assert len(manager.hooks) == 3  # model, layer1, layer2

    # Trigger the hooks
    model(torch.randn(1, 10))
    assert len(tensor_processor.processed_data) == 3

    manager.detach_hooks()
    assert len(manager.hooks) == 0

    # Ensure hooks are no longer active
    tensor_processor.processed_data = []
    model(torch.randn(1, 10))
    assert len(tensor_processor.processed_data) == 0


def test_should_instrument():
    manager = OptimizedHookManager(MagicMock())

    # Test FULL_TENSOR
    config = HookConfig(granularity=HookGranularity.FULL_TENSOR)
    assert manager._should_instrument("any_layer", config) is True

    # Test ATTENTION_ONLY
    config = HookConfig(granularity=HookGranularity.ATTENTION_ONLY)
    assert manager._should_instrument("decoder.block.0.attn", config) is True
    assert manager._should_instrument("decoder.block.0.mlp", config) is False

    # Test MLP_ONLY
    config = HookConfig(granularity=HookGranularity.MLP_ONLY)
    assert manager._should_instrument("decoder.block.0.attn", config) is False
    assert manager._should_instrument("decoder.block.0.mlp", config) is True

    # Test target_layers
    config = HookConfig(
        granularity=HookGranularity.FULL_TENSOR, target_layers=["layer1"]
    )
    assert manager._should_instrument("layer1", config) is True
    assert manager._should_instrument("layer2", config) is False


def test_hook_synchronous_processing():
    model = SimpleModel()
    tensor_processor = MockTensorProcessor()
    manager = OptimizedHookManager(tensor_processor)
    config = HookConfig(granularity=HookGranularity.FULL_TENSOR, async_enabled=False)

    manager.attach_hooks(model, config)

    input_tensor = torch.randn(1, 10)
    model(input_tensor)

    assert len(tensor_processor.processed_data) == 3
    # Check metadata from one of the calls
    _, metadata = tensor_processor.processed_data[0]
    assert metadata["sync"] is True
    assert "layer_name" in metadata


@patch(
    "llm_instrumentation.core.hooks.OptimizedHookManager." "_enqueue_async_processing"
)
def test_hook_asynchronous_processing(mock_enqueue):
    model = SimpleModel()
    tensor_processor = MockTensorProcessor()
    manager = OptimizedHookManager(tensor_processor)
    config = HookConfig(granularity=HookGranularity.FULL_TENSOR, async_enabled=True)

    manager.attach_hooks(model, config)

    input_tensor = torch.randn(1, 10)
    model(input_tensor)

    # Async queue should be called, direct processing should not
    assert mock_enqueue.called
    assert len(tensor_processor.processed_data) == 0
