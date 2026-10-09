from types import SimpleNamespace
from typing import Tuple

import pytest
import torch

import deep_ep
import deep_ep._C as _C
from deep_ep.buffers.elastic import ElasticBuffer
from deep_ep.buffers.legacy import Buffer
from deep_ep.buffers.token_drop import TokenDropHandle


def _reference(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = topk_idx.reshape(-1)
    weights = topk_weights.reshape(-1)
    weight_order = torch.argsort(weights, descending=True, stable=True)
    expert_order = torch.argsort(ids[weight_order], stable=True)
    order = weight_order[expert_order]
    sorted_ids = ids[order]
    positions = torch.arange(ids.numel(), device=ids.device)
    starts = torch.where(
        torch.cat((torch.ones(1, dtype=torch.bool, device=ids.device),
                   sorted_ids[1:] != sorted_ids[:-1])),
        positions,
        0,
    ).cummax(dim=0).values
    keep_sorted = (sorted_ids >= 0) & (positions - starts < expert_capacity)
    keep = torch.empty_like(keep_sorted)
    keep.scatter_(0, order, keep_sorted)
    discard = ~keep.reshape_as(topk_idx)
    return (
        topk_idx.masked_fill(discard, -1),
        topk_weights.masked_fill(discard, 0),
        discard.all(dim=1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("expert_capacity", [0, 1, 2, 8])
def test_native_route_selection_matches_weight_priority_reference(expert_capacity: int):
    topk_idx = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, -1]],
        device="cuda",
        dtype=deep_ep.topk_idx_t,
    )
    topk_weights = torch.tensor(
        [[0.1, 0.2], [0.9, 0.1], [0.9, 0.8], [0.05, 1.0]],
        device="cuda",
    )
    original_idx = topk_idx.clone()
    original_weights = topk_weights.clone()

    actual = _C._drop_tokens(topk_idx, topk_weights, expert_capacity)
    expected = _reference(topk_idx, topk_weights, expert_capacity)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    torch.testing.assert_close(topk_idx, original_idx)
    torch.testing.assert_close(topk_weights, original_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_native_route_selection_handles_empty_inputs():
    topk_idx = torch.empty((0, 2), device="cuda", dtype=deep_ep.topk_idx_t)
    topk_weights = torch.empty((0, 2), device="cuda")
    output_idx, output_weights, dropped_rows = _C._drop_tokens(
        topk_idx, topk_weights, 1)
    assert output_idx.shape == topk_idx.shape
    assert output_weights.shape == topk_weights.shape
    assert dropped_rows.shape == (0,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_native_route_selection_matches_random_routes():
    torch.manual_seed(0)
    topk_idx = torch.randint(
        -1, 17, (257, 4), device="cuda", dtype=deep_ep.topk_idx_t)
    topk_weights = torch.rand((257, 4), device="cuda")
    topk_weights[::7] = 0.5

    for expert_capacity in (1, 3, 16):
        actual = _C._drop_tokens(topk_idx, topk_weights, expert_capacity)
        expected = _reference(topk_idx, topk_weights, expert_capacity)
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)


class _LegacyRuntime:

    def __init__(self) -> None:
        self.capacity_dispatch_inputs = []
        self.capacity_combine_calls = 0

    def get_num_rdma_ranks(self):
        return 1

    def intranode_dispatch(self, x, x_scales, topk_idx, topk_weights, *args):
        tokens = x.size(0)
        return (
            x, x_scales, topk_idx, topk_weights, [tokens],
            torch.empty((1, 1), dtype=torch.int32),
            torch.empty((1, 1), dtype=torch.int32),
            torch.empty((1, 1), dtype=torch.int32),
            torch.arange(tokens, dtype=torch.int32),
            torch.empty((tokens, 1), dtype=torch.int32), None,
        )

    def intranode_combine(self, x, *args):
        return x, None, None

    def intranode_dispatch_with_capacity(self, x, x_scales, topk_idx,
                                         topk_weights, num_experts, capacity,
                                         *args):
        self.capacity_dispatch_inputs.append((x, topk_idx, topk_weights, capacity))
        kept = x[1:2]
        return (
            kept, x_scales, topk_idx[1:2], topk_weights[1:2], [1],
            torch.empty((1, 1), dtype=torch.int32),
            torch.empty((1, 1), dtype=torch.int32),
            torch.empty((1, 1), dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.empty((1, 1), dtype=torch.int32), None,
            torch.ones((1, 1), dtype=torch.bool),
            SimpleNamespace(capacity=capacity),
        )

    def intranode_combine_with_capacity(self, x, *args):
        self.capacity_combine_calls += 1
        return (
            torch.tensor([[0, 0], [2, 3], [0, 0]], dtype=x.dtype),
            None,
            None,
        )

    def low_latency_dispatch(self, x, topk_idx, *args):
        return x, None, torch.tensor([x.size(0)]), object(), object(), None, None

    def low_latency_dispatch_with_capacity(self, x, topk_idx, topk_weights,
                                           *args):
        self.capacity_dispatch_inputs.append((x, topk_idx, topk_weights, args[4]))
        return (x[1:2], None, torch.tensor([1]), object(), object(), None,
                None, SimpleNamespace(capacity=args[4],
                                      packed_tokens_per_rank=args[4]))

    def low_latency_combine(self, x, topk_idx, topk_weights, *args):
        return x, None, None

    def low_latency_combine_with_capacity(self, x, *args):
        self.capacity_combine_calls += 1
        return torch.tensor([[0, 0], [2, 3], [0, 0]], dtype=x.dtype), None, None

    def get_next_low_latency_combine_buffer(self, *args):
        self.next_low_latency_combine_buffer_args = args
        return args


def _legacy_buffer(runtime: _LegacyRuntime) -> Buffer:
    buffer = object.__new__(Buffer)
    buffer.group_size = 1
    buffer.runtime = runtime
    buffer.get_dispatch_config = lambda _: object()
    buffer.get_combine_config = lambda _: object()
    buffer.nvshmem_qp_depth = 8
    return buffer


def test_legacy_capacity_uses_native_dispatch_and_combine(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    runtime = _LegacyRuntime()
    buffer = _legacy_buffer(runtime)
    x = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
    topk_idx = torch.zeros((3, 1), dtype=deep_ep.topk_idx_t)
    topk_weights = torch.tensor([[0.1], [0.9], [0.2]])

    _, _, _, _, handle, _ = buffer.dispatch(
        x,
        num_tokens_per_expert=torch.full((1,), 3, dtype=torch.int32),
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        expert_capacity=1,
    )
    assert isinstance(handle, TokenDropHandle)
    assert runtime.capacity_dispatch_inputs == [(x, topk_idx, topk_weights, 1)]

    combined, _, _ = buffer.combine(x[1:2], handle, expert_capacity=1)
    torch.testing.assert_close(
        combined, torch.tensor([[0, 0], [2, 3], [0, 0]], dtype=x.dtype))
    assert runtime.capacity_combine_calls == 1


def test_legacy_without_capacity_preserves_original_path():
    runtime = _LegacyRuntime()
    buffer = _legacy_buffer(runtime)
    x = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
    topk_idx = torch.zeros((3, 1), dtype=deep_ep.topk_idx_t)
    topk_weights = torch.tensor([[0.1], [0.9], [0.2]])

    _, _, _, _, handle, _ = buffer.dispatch(
        x,
        num_tokens_per_rank=torch.ones(1, dtype=torch.int32),
        is_token_in_rank=torch.ones((3, 1), dtype=torch.bool),
        num_tokens_per_expert=torch.full((1,), 3, dtype=torch.int32),
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )
    assert isinstance(handle, tuple)
    combined, _, _ = buffer.combine(x, handle)
    torch.testing.assert_close(combined, x)
    assert not runtime.capacity_dispatch_inputs
    assert not runtime.capacity_combine_calls


def test_low_latency_capacity_uses_native_dispatch_and_combine(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    runtime = _LegacyRuntime()
    buffer = _legacy_buffer(runtime)
    x = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
    topk_idx = torch.zeros((3, 1), dtype=deep_ep.topk_idx_t)
    topk_weights = torch.tensor([[0.1], [0.9], [0.2]])

    _, _, handle, _, _ = buffer.low_latency_dispatch(
        x, topk_idx, num_max_dispatch_tokens_per_rank=3, num_experts=1,
        expert_capacity=1, topk_weights=topk_weights)
    assert isinstance(handle, TokenDropHandle)
    output = torch.empty_like(x)
    combined, _, _ = buffer.low_latency_combine(
        x[1:2], topk_idx, topk_weights, handle, out=output,
        expert_capacity=1)
    torch.testing.assert_close(
        combined, torch.tensor([[0, 0], [2, 3], [0, 0]], dtype=x.dtype))
    assert runtime.capacity_combine_calls == 1

    raw_buffer = buffer.get_next_low_latency_combine_buffer(handle)
    assert raw_buffer == (3, 1, 2, 1)


class _ElasticRuntime:

    def __init__(self) -> None:
        self.capacity_dispatch_inputs = []
        self.capacity_combine_calls = 0

    @staticmethod
    def _dispatch_result(x, sf, topk_idx, topk_weights):
        tokens = x.size(0)
        return (
            x, sf, topk_idx, topk_weights, topk_idx.clone(), [tokens],
            torch.tensor([tokens]), torch.tensor([tokens]),
            torch.arange(tokens).unsqueeze(1),
            torch.zeros(tokens, dtype=torch.int32), None, None, None,
        )

    def dispatch(self, x, sf, topk_idx, topk_weights, *args):
        return self._dispatch_result(x, sf, topk_idx, topk_weights)

    def dispatch_with_capacity(self, x, sf, topk_idx, topk_weights, *args):
        self.capacity_dispatch_inputs.append((x, topk_idx, topk_weights, args[-1]))
        return (*self._dispatch_result(x[1:2], sf, topk_idx[1:2], topk_weights[1:2]),
                SimpleNamespace(capacity=args[-1]))

    def combine(self, x, *args):
        return x, None, None

    def combine_with_capacity(self, x, *args):
        self.capacity_combine_calls += 1
        return torch.tensor([[0, 0], [2, 3], [0, 0]], dtype=x.dtype), None, None


def test_elastic_capacity_uses_native_dispatch_and_combine(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    runtime = _ElasticRuntime()
    buffer = object.__new__(ElasticBuffer)
    buffer.runtime = runtime
    buffer.num_allocated_qps = 1
    buffer.num_max_tokens_per_rank = 3
    buffer.get_theoretical_num_sms = lambda *_: 1
    buffer.get_theoretical_num_qps = lambda *_: 1
    x = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
    topk_idx = torch.zeros((3, 1), dtype=deep_ep.topk_idx_t)
    topk_weights = torch.tensor([[0.1], [0.9], [0.2]])

    _, _, _, handle, _ = buffer.dispatch(
        x, topk_idx=topk_idx, topk_weights=topk_weights, num_experts=1,
        num_max_tokens_per_rank=3, expert_capacity=1)
    assert isinstance(handle, TokenDropHandle)
    assert runtime.capacity_dispatch_inputs == [(x, topk_idx, topk_weights, 1)]

    combined, _, _ = buffer.combine(x[1:2], handle, expert_capacity=1)
    torch.testing.assert_close(
        combined, torch.tensor([[0, 0], [2, 3], [0, 0]], dtype=x.dtype))
    assert runtime.capacity_combine_calls == 1
