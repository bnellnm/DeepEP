#pragma once

#include <torch/extension.h>

#include <memory>
#include <optional>
#include <tuple>

namespace deep_ep {

struct TokenDropState {
    int capacity;
    int64_t original_num_tokens;
    int num_retained_tokens;
    torch::Tensor retained_rows;
    torch::Tensor topk_idx;
    torch::Tensor topk_weights;
    torch::Tensor x;
    std::optional<torch::Tensor> x_scales;
    torch::Tensor original_topk_idx;
    torch::Tensor selected_topk_idx;
    int packed_tokens_per_rank = 0;
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> drop_tokens(const torch::Tensor& topk_idx,
                                                                    const torch::Tensor& topk_weights,
                                                                    int expert_capacity);

std::shared_ptr<TokenDropState> make_token_drop_state(const torch::Tensor& x,
                                                      const std::optional<torch::Tensor>& x_scales,
                                                      const torch::Tensor& topk_idx,
                                                      const torch::Tensor& topk_weights,
                                                      int expert_capacity);

std::shared_ptr<TokenDropState> make_low_latency_token_drop_state(const torch::Tensor& x,
                                                                  const torch::Tensor& topk_idx,
                                                                  const torch::Tensor& topk_weights,
                                                                  int expert_capacity,
                                                                  int num_max_dispatch_tokens_per_rank);

torch::Tensor compact_token_drop_rows(const torch::Tensor& input, const TokenDropState& state);

std::optional<torch::Tensor> compact_token_drop_rows(const std::optional<torch::Tensor>& input, const TokenDropState& state);

torch::Tensor scatter_token_drop_rows(const torch::Tensor& input,
                                      const TokenDropState& state,
                                      const std::optional<torch::Tensor>& out = std::nullopt);

}  // namespace deep_ep
