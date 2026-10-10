from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TokenDropStats:
    num_tokens: int
    num_routable_tokens: int
    num_input_assignments: int
    num_kept_assignments: int
    num_dropped_assignments: int
    num_fully_dropped_tokens: int
    num_partially_dropped_tokens: int

    @property
    def assignment_drop_rate(self) -> float:
        if self.num_input_assignments == 0:
            return 0.0
        return self.num_dropped_assignments / self.num_input_assignments

    @property
    def fully_dropped_token_rate(self) -> float:
        if self.num_routable_tokens == 0:
            return 0.0
        return self.num_fully_dropped_tokens / self.num_routable_tokens


@dataclass
class TokenDropHandle:
    """Opaque native token-dropping state paired with a DeepEP handle."""

    handle: Any
    state: Any

    def __getattr__(self, name: str) -> Any:
        return getattr(self.handle, name)

    def get_token_drop_stats(self) -> TokenDropStats:
        """Return token-drop statistics, synchronizing with the device."""
        original_valid = self.state.original_topk_idx >= 0
        selected_valid = self.state.selected_topk_idx >= 0
        kept = original_valid & selected_valid
        input_per_token = original_valid.sum(dim=1)
        kept_per_token = kept.sum(dim=1)
        routable = input_per_token > 0
        dropped_per_token = input_per_token - kept_per_token

        counts = torch.stack(
            (
                routable.sum(),
                original_valid.sum(),
                kept.sum(),
                (routable & (kept_per_token == 0)).sum(),
                ((kept_per_token > 0) & (dropped_per_token > 0)).sum(),
            )
        ).tolist()
        (
            num_routable_tokens,
            num_input_assignments,
            num_kept_assignments,
            num_fully_dropped_tokens,
            num_partially_dropped_tokens,
        ) = counts
        return TokenDropStats(
            num_tokens=self.state.original_topk_idx.size(0),
            num_routable_tokens=num_routable_tokens,
            num_input_assignments=num_input_assignments,
            num_kept_assignments=num_kept_assignments,
            num_dropped_assignments=num_input_assignments - num_kept_assignments,
            num_fully_dropped_tokens=num_fully_dropped_tokens,
            num_partially_dropped_tokens=num_partially_dropped_tokens,
        )
