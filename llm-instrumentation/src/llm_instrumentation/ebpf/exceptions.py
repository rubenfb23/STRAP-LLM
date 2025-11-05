"""Custom exceptions used by the ebpf package."""


class EBPFError(RuntimeError):
    """Base class for instrumentation errors."""


class BPFUnavailableError(EBPFError):
    """Raised when the BCC runtime or tracepoints are missing."""


class CollectorAttachError(EBPFError):
    """Raised when collectors fail to attach their probes."""


class CollectorRuntimeError(EBPFError):
    """Raised when collectors encounter runtime errors consuming data."""
