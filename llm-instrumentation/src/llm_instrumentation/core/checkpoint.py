import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CheckpointState:
    """Represents the data persisted inside a checkpoint file."""

    tokens: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    total_tokens: int


class CheckpointManager:
    """Persist incremental token snapshots and metadata for long-running captures.

    The checkpoint file keeps a full copy of the token history to make resume logic
    straightforward. Snapshots are rewritten atomically using temporary files to
    avoid partial writes if the process exits unexpectedly.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        path: str,
        interval_tokens: Optional[int],
        resume: bool = False,
    ) -> None:
        self.path = Path(path)
        self.interval_tokens = (
            max(1, int(interval_tokens)) if interval_tokens else None
        )
        self.resume = resume
        self._last_saved_total = 0
        self._cached_state: Optional[CheckpointState] = None

        if self.resume:
            self._cached_state = self._load_state()
            if self._cached_state:
                self._last_saved_total = self._cached_state.total_tokens

    def has_interval(self) -> bool:
        """Return True if checkpointing is enabled."""
        return self.interval_tokens is not None

    def load(self) -> CheckpointState:
        """Load the checkpoint state from disk."""
        if self._cached_state:
            return self._cached_state
        state = self._load_state()
        if not state:
            state = CheckpointState(tokens=[], metadata={}, total_tokens=0)
        self._cached_state = state
        return state

    def _load_state(self) -> Optional[CheckpointState]:
        if not self.path.exists():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        tokens = data.get("tokens", [])
        metadata = data.get("metadata", {})
        total_tokens = data.get("total_tokens", len(tokens))
        return CheckpointState(tokens=tokens, metadata=metadata, total_tokens=total_tokens)

    def should_checkpoint(self, total_tokens: int) -> bool:
        if not self.interval_tokens:
            return False
        return total_tokens - self._last_saved_total >= self.interval_tokens

    def maybe_checkpoint(
        self,
        tokens: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> bool:
        """Persist a checkpoint if the interval has been exceeded."""
        total_tokens = len(tokens)
        if not self.should_checkpoint(total_tokens):
            return False
        self._write_state(tokens, metadata)
        return True

    def finalize(
        self,
        tokens: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        keep_file: bool = False,
    ) -> None:
        """Persist the final snapshot and optionally clean up the checkpoint file."""
        if tokens:
            self._write_state(tokens, metadata)
        elif self.path.exists():
            # Ensure metadata-only captures still flush state
            self._write_state(tokens, metadata)

        if not keep_file and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass

    def _write_state(
        self,
        tokens: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        tokens_copy = [dict(token) for token in tokens]
        metadata_copy = dict(metadata)

        payload = {
            "version": self.CHECKPOINT_VERSION,
            "total_tokens": len(tokens),
            "last_checkpoint_time": time.time(),
            "metadata": metadata_copy,
            "tokens": tokens_copy,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=str(self.path.parent)
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False)
            tmp_path = Path(tmp.name)

        try:
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        self._last_saved_total = len(tokens_copy)
        self._cached_state = CheckpointState(
            tokens=tokens_copy,
            metadata=metadata_copy,
            total_tokens=len(tokens_copy),
        )
