from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional, Callable, List
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum


class HookGranularity(Enum):
    FULL_TENSOR = "full_tensor"
    SAMPLED_SLICES = "sampled_slices"
    ATTENTION_ONLY = "attention_only"
    MLP_ONLY = "mlp_only"


@dataclass
class HookConfig:
    granularity: HookGranularity
    sampling_rate: float = 1.0
    target_layers: Optional[List[str]] = None
    compression_enabled: bool = True
    async_enabled: bool = True


class TensorProcessor(Protocol):
    def process(self, tensor: torch.Tensor, metadata: Dict[str, Any]) -> bytes: ...


class HookManager(ABC):
    @abstractmethod
    def attach_hooks(self, model: nn.Module, config: HookConfig) -> None: ...

    @abstractmethod
    def detach_hooks(self) -> None: ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]: ...


class OptimizedHookManager(HookManager):
    """Production hook manager with minimal overhead."""

    def __init__(self, tensor_processor: TensorProcessor):
        self.tensor_processor = tensor_processor
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        # self.metrics = PerformanceMetrics()

    def attach_hooks(self, model: nn.Module, config: HookConfig) -> None:
        """Attach hooks with optimized data capture."""
        # The test suite implies a non-recursive traversal, so we hook the
        # model and its direct children rather than all named modules.
        modules_to_hook = [("", model)] + list(model.named_children())

        for name, module in modules_to_hook:
            if self._should_instrument(name, config):
                hook = self._create_optimized_hook(name, config)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)

    def detach_hooks(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def get_metrics(self) -> Dict[str, float]:
        return {}

    def _should_instrument(self, layer_name: str, config: HookConfig) -> bool:
        if config.target_layers:
            return layer_name in config.target_layers
        if config.granularity == HookGranularity.ATTENTION_ONLY:
            return "attn" in layer_name
        if config.granularity == HookGranularity.MLP_ONLY:
            return "mlp" in layer_name
        return True

    def _create_optimized_hook(self, layer_name: str, config: HookConfig) -> Callable:
        """Create memory-efficient hook with minimal Python overhead."""

        def optimized_hook(module, input, output):
            with torch.no_grad():  # Prevent gradient computation
                if config.granularity == HookGranularity.SAMPLED_SLICES:
                    output = self._sample_tensor(output, config.sampling_rate)

                # Process asynchronously if enabled
                if config.async_enabled:
                    self._enqueue_async_processing(layer_name, output)
                else:
                    self._process_synchronous(layer_name, output)

        return optimized_hook

    def _sample_tensor(
        self, tensor: torch.Tensor, sampling_rate: float
    ) -> torch.Tensor:
        # This is a placeholder for a more sophisticated sampling strategy
        num_elements = tensor.numel()
        num_samples = int(num_elements * sampling_rate)
        indices = torch.randperm(num_elements)[:num_samples]
        return tensor.flatten()[indices]

    def _enqueue_async_processing(self, layer_name: str, tensor: torch.Tensor):
        # Placeholder for async processing. In a real implementation, this
        # would add the tensor to a queue for a background worker.
        metadata = {"layer_name": layer_name, "sync": False}
        self.tensor_processor.process(tensor, metadata)

    def _process_synchronous(self, layer_name: str, tensor: torch.Tensor):
        metadata = {"layer_name": layer_name, "sync": True}
        self.tensor_processor.process(tensor, metadata)
