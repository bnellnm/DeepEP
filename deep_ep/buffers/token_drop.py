from dataclasses import dataclass
from typing import Any


@dataclass
class TokenDropHandle:
    """Opaque native token-dropping state paired with a DeepEP handle."""

    handle: Any
    state: Any

    def __getattr__(self, name: str) -> Any:
        return getattr(self.handle, name)
