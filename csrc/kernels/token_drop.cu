#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <limits>

#include "token_drop.hpp"

namespace deep_ep {
namespace {

constexpr int kNumThreads = 256;

struct TokenDropSelection {
    torch::Tensor topk_idx;
    torch::Tensor topk_weights;
    torch::Tensor retained_rows;
    int num_retained_rows;
};

__global__ void initialize_indices_kernel(int* indices, int num_assignments) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_assignments)
        indices[idx] = idx;
}

__global__ void gather_expert_keys_kernel(const topk_idx_t* topk_idx, const int* indices, topk_idx_t* expert_keys, int num_assignments) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_assignments)
        expert_keys[idx] = topk_idx[indices[idx]];
}

__global__ void finalize_assignments_kernel(const topk_idx_t* topk_idx,
                                            const float* topk_weights,
                                            const int* sorted_indices,
                                            const int* ranks,
                                            topk_idx_t* output_topk_idx,
                                            float* output_topk_weights,
                                            int num_assignments,
                                            int expert_capacity) {
    const int sorted_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_idx >= num_assignments)
        return;

    const int original_idx = sorted_indices[sorted_idx];
    const topk_idx_t expert_idx = topk_idx[original_idx];
    const bool keep = expert_idx >= 0 && ranks[sorted_idx] <= expert_capacity;
    output_topk_idx[original_idx] = keep ? expert_idx : -1;
    output_topk_weights[original_idx] = keep ? topk_weights[original_idx] : 0.0f;
}

__global__ void mark_dropped_rows_kernel(const topk_idx_t* topk_idx, bool* dropped_rows, int num_tokens, int num_topk) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens)
        return;

    bool has_kept_assignment = false;
    for (int slot_idx = threadIdx.x; slot_idx < num_topk; slot_idx += blockDim.x)
        has_kept_assignment |= topk_idx[token_idx * num_topk + slot_idx] >= 0;

    __shared__ int row_has_kept_assignment;
    if (threadIdx.x == 0)
        row_has_kept_assignment = 0;
    __syncthreads();
    if (has_kept_assignment)
        atomicExch(&row_has_kept_assignment, 1);
    __syncthreads();
    if (threadIdx.x == 0)
        dropped_rows[token_idx] = row_has_kept_assignment == 0;
}

__global__ void mark_retained_rows_kernel(const int* retained_rows, int num_retained_rows, bool* dropped_rows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_retained_rows)
        dropped_rows[retained_rows[idx]] = false;
}

__global__ void compact_rows_kernel(
    const uint8_t* input, uint8_t* output, const int* retained_rows, int num_retained_rows, int64_t row_bytes) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t num_bytes = static_cast<int64_t>(num_retained_rows) * row_bytes;
    if (idx < num_bytes) {
        const int row = static_cast<int>(idx / row_bytes);
        const int64_t offset = idx % row_bytes;
        output[idx] = input[static_cast<int64_t>(retained_rows[row]) * row_bytes + offset];
    }
}

__global__ void scatter_rows_kernel(
    const uint8_t* input, uint8_t* output, const int* retained_rows, int num_retained_rows, int64_t row_bytes) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t num_bytes = static_cast<int64_t>(num_retained_rows) * row_bytes;
    if (idx < num_bytes) {
        const int row = static_cast<int>(idx / row_bytes);
        const int64_t offset = idx % row_bytes;
        output[static_cast<int64_t>(retained_rows[row]) * row_bytes + offset] = input[idx];
    }
}

TokenDropSelection select_token_drop_routes(const torch::Tensor& topk_idx, const torch::Tensor& topk_weights, int expert_capacity) {
    EP_HOST_ASSERT(expert_capacity >= 0);
    EP_HOST_ASSERT(topk_idx.is_cuda() && topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_weights.is_cuda() && topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_idx.dim() == 2 && topk_weights.dim() == 2);
    EP_HOST_ASSERT(topk_idx.sizes() == topk_weights.sizes());
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat);
    EP_HOST_ASSERT(topk_idx.device() == topk_weights.device());
    EP_HOST_ASSERT(topk_idx.size(0) <= std::numeric_limits<int>::max());
    EP_HOST_ASSERT(topk_idx.size(1) <= std::numeric_limits<int>::max());
    EP_HOST_ASSERT(topk_idx.numel() <= std::numeric_limits<int>::max());

    const int num_tokens = static_cast<int>(topk_idx.size(0));
    const int num_topk = static_cast<int>(topk_idx.size(1));
    auto output_topk_idx = torch::empty_like(topk_idx);
    auto output_topk_weights = torch::empty_like(topk_weights);
    auto retained_rows = torch::empty({0}, topk_idx.options().dtype(torch::kInt));

    if (num_tokens == 0 || num_topk == 0)
        return {output_topk_idx, output_topk_weights, retained_rows, 0};

    const int num_assignments = num_tokens * num_topk;
    const auto stream = at::cuda::getCurrentCUDAStream(topk_idx.get_device());
    const int num_blocks = (num_assignments + kNumThreads - 1) / kNumThreads;

    auto sorted_weights = torch::empty_like(topk_weights).view({num_assignments});
    auto expert_keys_in = torch::empty_like(topk_idx).view({num_assignments});
    auto expert_keys_out = torch::empty_like(topk_idx).view({num_assignments});
    auto indices_in = torch::empty({num_assignments}, topk_idx.options().dtype(torch::kInt));
    auto indices_out = torch::empty_like(indices_in);
    auto ranks = torch::empty_like(indices_in);
    auto ones = torch::ones_like(indices_in);

    initialize_indices_kernel<<<num_blocks, kNumThreads, 0, stream>>>(indices_in.data_ptr<int>(), num_assignments);
    CUDA_RUNTIME_CHECK(cudaGetLastError());

    size_t sort_weights_temp_bytes = 0;
    CUDA_RUNTIME_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                                                 sort_weights_temp_bytes,
                                                                 topk_weights.data_ptr<float>(),
                                                                 sorted_weights.data_ptr<float>(),
                                                                 indices_in.data_ptr<int>(),
                                                                 indices_out.data_ptr<int>(),
                                                                 num_assignments,
                                                                 0,
                                                                 sizeof(float) * 8,
                                                                 stream));

    size_t sort_experts_temp_bytes = 0;
    CUDA_RUNTIME_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,
                                                       sort_experts_temp_bytes,
                                                       expert_keys_in.data_ptr<topk_idx_t>(),
                                                       expert_keys_out.data_ptr<topk_idx_t>(),
                                                       indices_out.data_ptr<int>(),
                                                       indices_in.data_ptr<int>(),
                                                       num_assignments,
                                                       0,
                                                       sizeof(topk_idx_t) * 8,
                                                       stream));

    size_t scan_temp_bytes = 0;
    CUDA_RUNTIME_CHECK(cub::DeviceScan::InclusiveSumByKey(nullptr,
                                                          scan_temp_bytes,
                                                          expert_keys_out.data_ptr<topk_idx_t>(),
                                                          ones.data_ptr<int>(),
                                                          ranks.data_ptr<int>(),
                                                          num_assignments,
                                                          ::cuda::std::equal_to<topk_idx_t>(),
                                                          stream));

    const size_t temp_bytes = std::max(sort_weights_temp_bytes, std::max(sort_experts_temp_bytes, scan_temp_bytes));
    auto temp_storage = torch::empty({static_cast<int64_t>(temp_bytes)}, topk_idx.options().dtype(torch::kByte));

    CUDA_RUNTIME_CHECK(cub::DeviceRadixSort::SortPairsDescending(temp_storage.data_ptr(),
                                                                 sort_weights_temp_bytes,
                                                                 topk_weights.data_ptr<float>(),
                                                                 sorted_weights.data_ptr<float>(),
                                                                 indices_in.data_ptr<int>(),
                                                                 indices_out.data_ptr<int>(),
                                                                 num_assignments,
                                                                 0,
                                                                 sizeof(float) * 8,
                                                                 stream));
    gather_expert_keys_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
        topk_idx.data_ptr<topk_idx_t>(), indices_out.data_ptr<int>(), expert_keys_in.data_ptr<topk_idx_t>(), num_assignments);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cub::DeviceRadixSort::SortPairs(temp_storage.data_ptr(),
                                                       sort_experts_temp_bytes,
                                                       expert_keys_in.data_ptr<topk_idx_t>(),
                                                       expert_keys_out.data_ptr<topk_idx_t>(),
                                                       indices_out.data_ptr<int>(),
                                                       indices_in.data_ptr<int>(),
                                                       num_assignments,
                                                       0,
                                                       sizeof(topk_idx_t) * 8,
                                                       stream));
    CUDA_RUNTIME_CHECK(cub::DeviceScan::InclusiveSumByKey(temp_storage.data_ptr(),
                                                          scan_temp_bytes,
                                                          expert_keys_out.data_ptr<topk_idx_t>(),
                                                          ones.data_ptr<int>(),
                                                          ranks.data_ptr<int>(),
                                                          num_assignments,
                                                          ::cuda::std::equal_to<topk_idx_t>(),
                                                          stream));

    finalize_assignments_kernel<<<num_blocks, kNumThreads, 0, stream>>>(topk_idx.data_ptr<topk_idx_t>(),
                                                                        topk_weights.data_ptr<float>(),
                                                                        indices_in.data_ptr<int>(),
                                                                        ranks.data_ptr<int>(),
                                                                        output_topk_idx.data_ptr<topk_idx_t>(),
                                                                        output_topk_weights.data_ptr<float>(),
                                                                        num_assignments,
                                                                        expert_capacity);
    CUDA_RUNTIME_CHECK(cudaGetLastError());

    auto retained_flags = torch::empty({num_tokens}, topk_idx.options().dtype(torch::kBool));
    mark_dropped_rows_kernel<<<num_tokens, kNumThreads, 0, stream>>>(
        output_topk_idx.data_ptr<topk_idx_t>(), retained_flags.data_ptr<bool>(), num_tokens, num_topk);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    retained_flags.logical_not_();

    auto row_indices = torch::arange(num_tokens, topk_idx.options().dtype(torch::kInt));
    auto retained_count_device = torch::empty({1}, topk_idx.options().dtype(torch::kInt));
    size_t select_temp_bytes = 0;
    CUDA_RUNTIME_CHECK(cub::DeviceSelect::Flagged(nullptr,
                                                  select_temp_bytes,
                                                  row_indices.data_ptr<int>(),
                                                  retained_flags.data_ptr<bool>(),
                                                  static_cast<int*>(nullptr),
                                                  retained_count_device.data_ptr<int>(),
                                                  num_tokens,
                                                  stream));
    auto select_temp_storage = torch::empty({static_cast<int64_t>(select_temp_bytes)}, topk_idx.options().dtype(torch::kByte));
    auto retained_rows_full = torch::empty({num_tokens}, topk_idx.options().dtype(torch::kInt));
    CUDA_RUNTIME_CHECK(cub::DeviceSelect::Flagged(select_temp_storage.data_ptr(),
                                                  select_temp_bytes,
                                                  row_indices.data_ptr<int>(),
                                                  retained_flags.data_ptr<bool>(),
                                                  retained_rows_full.data_ptr<int>(),
                                                  retained_count_device.data_ptr<int>(),
                                                  num_tokens,
                                                  stream));

    int num_retained_rows = 0;
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        &num_retained_rows, retained_count_device.data_ptr<int>(), sizeof(num_retained_rows), cudaMemcpyDeviceToHost, stream));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    retained_rows = retained_rows_full.narrow(0, 0, num_retained_rows);
    return {output_topk_idx, output_topk_weights, retained_rows, num_retained_rows};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> drop_tokens(const torch::Tensor& topk_idx,
                                                                    const torch::Tensor& topk_weights,
                                                                    int expert_capacity) {
    auto selection = select_token_drop_routes(topk_idx, topk_weights, expert_capacity);
    auto dropped_rows = torch::ones({topk_idx.size(0)}, topk_idx.options().dtype(torch::kBool));
    if (selection.num_retained_rows > 0) {
        const auto stream = at::cuda::getCurrentCUDAStream(topk_idx.get_device());
        const int num_blocks = (selection.num_retained_rows + kNumThreads - 1) / kNumThreads;
        mark_retained_rows_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
            selection.retained_rows.data_ptr<int>(), selection.num_retained_rows, dropped_rows.data_ptr<bool>());
        CUDA_RUNTIME_CHECK(cudaGetLastError());
    }
    return {selection.topk_idx, selection.topk_weights, dropped_rows};
}

torch::Tensor compact_token_drop_rows(const torch::Tensor& input, const TokenDropState& state) {
    EP_HOST_ASSERT(input.is_cuda() && input.is_contiguous());
    EP_HOST_ASSERT(input.dim() >= 1 && input.size(0) == state.original_num_tokens);
    auto shape = input.sizes().vec();
    shape[0] = state.num_retained_tokens;
    auto output = torch::empty(shape, input.options());
    if (state.num_retained_tokens == 0)
        return output;

    const int64_t row_bytes = input.nbytes() / state.original_num_tokens;
    const int64_t num_bytes = static_cast<int64_t>(state.num_retained_tokens) * row_bytes;
    EP_HOST_ASSERT(num_bytes <= std::numeric_limits<int>::max() * static_cast<int64_t>(kNumThreads));
    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    const int num_blocks = static_cast<int>((num_bytes + kNumThreads - 1) / kNumThreads);
    compact_rows_kernel<<<num_blocks, kNumThreads, 0, stream>>>(static_cast<const uint8_t*>(input.data_ptr()),
                                                                static_cast<uint8_t*>(output.data_ptr()),
                                                                state.retained_rows.data_ptr<int>(),
                                                                state.num_retained_tokens,
                                                                row_bytes);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return output;
}

std::optional<torch::Tensor> compact_token_drop_rows(const std::optional<torch::Tensor>& input, const TokenDropState& state) {
    if (!input.has_value())
        return std::nullopt;
    return compact_token_drop_rows(input.value(), state);
}

std::shared_ptr<TokenDropState> make_token_drop_state(const torch::Tensor& x,
                                                      const std::optional<torch::Tensor>& x_scales,
                                                      const torch::Tensor& topk_idx,
                                                      const torch::Tensor& topk_weights,
                                                      int expert_capacity) {
    EP_HOST_ASSERT(x.is_cuda() && x.is_contiguous());
    EP_HOST_ASSERT(x.dim() >= 1 && x.size(0) == topk_idx.size(0));
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x_scales->is_cuda());
        EP_HOST_ASSERT(x_scales->dim() >= 1 && x_scales->size(0) == x.size(0));
    }

    auto selection = select_token_drop_routes(topk_idx, topk_weights, expert_capacity);
    auto state = std::make_shared<TokenDropState>(TokenDropState{
        expert_capacity,
        x.size(0),
        selection.num_retained_rows,
        selection.retained_rows,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        std::nullopt,
    });
    state->topk_idx = compact_token_drop_rows(selection.topk_idx, *state);
    state->topk_weights = compact_token_drop_rows(selection.topk_weights, *state);
    state->x = compact_token_drop_rows(x, *state);
    state->x_scales =
        x_scales.has_value() ? std::optional<torch::Tensor>(compact_token_drop_rows(x_scales->contiguous(), *state)) : std::nullopt;
    return state;
}

torch::Tensor scatter_token_drop_rows(const torch::Tensor& input, const TokenDropState& state, const std::optional<torch::Tensor>& out) {
    EP_HOST_ASSERT(input.is_cuda() && input.is_contiguous());
    EP_HOST_ASSERT(input.dim() >= 1 && input.size(0) == state.num_retained_tokens);
    auto shape = input.sizes().vec();
    shape[0] = state.original_num_tokens;
    auto output = out.value_or(torch::empty(shape, input.options()));
    EP_HOST_ASSERT(output.is_cuda() && output.is_contiguous());
    EP_HOST_ASSERT(output.sizes().vec() == shape);
    EP_HOST_ASSERT(output.scalar_type() == input.scalar_type());

    const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(output.data_ptr(), 0, output.nbytes(), stream));
    if (state.num_retained_tokens == 0)
        return output;

    const int64_t row_bytes = input.nbytes() / state.num_retained_tokens;
    const int64_t num_bytes = static_cast<int64_t>(state.num_retained_tokens) * row_bytes;
    EP_HOST_ASSERT(num_bytes <= std::numeric_limits<int>::max() * static_cast<int64_t>(kNumThreads));
    const int num_blocks = static_cast<int>((num_bytes + kNumThreads - 1) / kNumThreads);
    scatter_rows_kernel<<<num_blocks, kNumThreads, 0, stream>>>(static_cast<const uint8_t*>(input.data_ptr()),
                                                                static_cast<uint8_t*>(output.data_ptr()),
                                                                state.retained_rows.data_ptr<int>(),
                                                                state.num_retained_tokens,
                                                                row_bytes);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return output;
}

}  // namespace deep_ep
