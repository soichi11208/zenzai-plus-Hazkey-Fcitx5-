#include "llama-context.h"
#include "llama-model.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {
thread_local std::string last_error;
void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

struct work_graph {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_backend_t backend;
    explicit work_graph(ggml_backend_t backend) : backend(backend) {
        ctx.reset(ggml_init({32*1024*1024, nullptr, true}));
        require(bool(ctx), "cannot allocate native TTT graph");
    }
    ggml_cgraph * graph() { return ggml_new_graph_custom(ctx.get(), 4096, false); }
    void allocate() {
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        require(bool(buffer), "cannot allocate native TTT tensors");
        ggml_backend_buffer_clear(buffer.get(), 0);
    }
    void run(ggml_cgraph * graph) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "native TTT graph failed");
    }
};

std::vector<llama_token> tokenize(llama_context * ctx, const char * text, bool eos) {
    require(text != nullptr, "missing TTT text");
    const auto * vocab = llama_model_get_vocab(&ctx->model);
    const auto size = std::strlen(text);
    require(size <= 64*1024, "TTT text is too long");
    int count = llama_tokenize(vocab, text, int(size), nullptr, 0, false, true);
    require(count <= 0, "unexpected tokenizer result");
    std::vector<llama_token> tokens(size_t(-count));
    count = llama_tokenize(vocab, text, int(size), tokens.data(), int(tokens.size()), false, true);
    require(count >= 0, "cannot tokenize TTT text");
    tokens.resize(size_t(count));
    if (eos) tokens.push_back(llama_vocab_eos(vocab));
    require(tokens.size() >= 1 && tokens.size() <= llama_n_ctx(ctx), "TTT input exceeds model context");
    return tokens;
}

void capture(llama_context * ctx, const std::vector<llama_token> & tokens, ggml_backend_t backend) {
    require(tokens.size() <= llama_n_ubatch(ctx), "TTT input exceeds physical batch size");
    llama_synchronize(ctx);
    llama_kv_cache_clear(ctx);
    if (ctx->ttt_capture_dest.empty() || ctx->ttt_capture_dest.front()->ne[1] != int64_t(tokens.size())) {
        ctx->ttt_capture_buffer.reset();
        ctx->ttt_capture_ctx.reset(ggml_init({1024*1024, nullptr, true}));
        require(bool(ctx->ttt_capture_ctx), "cannot allocate native TTT capture context");
        ctx->ttt_capture_dest.clear();
        for (const auto & layer : ctx->model.layers) {
            ctx->ttt_capture_dest.push_back(ggml_new_tensor_2d(ctx->ttt_capture_ctx.get(), GGML_TYPE_F32,
                layer.ffn_down->ne[0], tokens.size()));
        }
        ctx->ttt_residual_dest = ggml_new_tensor_2d(ctx->ttt_capture_ctx.get(), GGML_TYPE_F32,
            ctx->model.hparams.n_embd, tokens.size());
        ctx->ttt_capture_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx->ttt_capture_ctx.get(), backend));
        require(bool(ctx->ttt_capture_buffer), "cannot allocate native TTT capture buffer");
    }
    ctx->ttt_activations.assign(ctx->model.hparams.n_layer, nullptr);
    ctx->ttt_last_residual = nullptr;
    ctx->ttt_capture = true;
    auto batch = llama_batch_init(int(tokens.size()), 0, 1);
    batch.n_tokens = int(tokens.size());
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    const int result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    ctx->ttt_capture = false;
    llama_synchronize(ctx);
    require(result == 0, "native TTT forward pass failed");
    for (auto * z : ctx->ttt_activations) require(z != nullptr, "missing GPT-2 MLP activation");
    require(ctx->ttt_last_residual != nullptr, "missing GPT-2 final residual");
}

ggml_tensor * external(ggml_context * ctx, ggml_tensor * tensor) {
    auto * leaf = ggml_dup_tensor(ctx, tensor);
    leaf->data = tensor->data;
    leaf->buffer = tensor->buffer;
    std::copy(std::begin(tensor->nb), std::end(tensor->nb), std::begin(leaf->nb));
    return leaf;
}

ggml_tensor * scalar(ggml_context * ctx) { return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); }
void set_float(ggml_tensor * tensor, float value) { ggml_backend_tensor_set(tensor, &value, 0, sizeof(value)); }
float get_float(ggml_tensor * tensor) { float value; ggml_backend_tensor_get(tensor, &value, 0, sizeof(value)); return value; }
ggml_tensor * rows_mean(ggml_context * ctx, ggml_tensor * x) {
    return ggml_scale(ctx, ggml_sum_rows(ctx, x), 1.0f/float(x->ne[0]));
}
void require_norms(const std::vector<ggml_tensor *> & norms) {
    for (auto * norm : norms) {
        const float value = get_float(norm);
        require(std::isfinite(value) && value > 0, "invalid native TTT weights; update rejected");
    }
}
}

struct llama_ttt_state {
    std::shared_ptr<ggml_backend> backend;
    std::shared_ptr<llama_ttt_state> constants_owner;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    std::vector<ggml_tensor *> down;
    ggml_tensor * last_bias = nullptr;
    ggml_tensor * embedding = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * norm_bias = nullptr;
};

namespace {
llama_ttt_state & state_for(llama_context * ctx, bool write = false) {
    auto & model = const_cast<llama_model &>(ctx->model);
    require(model.arch == LLM_ARCH_GPT2, "native TTT requires GPT-2");
    if (model.ttt_state && (!write || model.ttt_state.use_count() == 1)) return *model.ttt_state;
    llama_synchronize(ctx);
    auto previous = model.ttt_state;
    auto state = std::make_shared<llama_ttt_state>();
    auto * first = model.layers.at(0).ffn_down;
    auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(first->buffer));
    if (previous) {
        state->backend = previous->backend;
        state->constants_owner = previous->constants_owner ? previous->constants_owner : previous;
    } else {
        state->backend = std::shared_ptr<ggml_backend>(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    }
    require(bool(state->backend), "cannot initialize native TTT backend");
    state->ctx.reset(ggml_init({4*1024*1024, nullptr, true}));
    require(bool(state->ctx), "cannot allocate native TTT weights");
    auto * init = ggml_new_graph_custom(state->ctx.get(), 1024, false);
    for (auto & layer : model.layers) {
        require(layer.ffn_down && layer.ffn_down_b, "GPT-2 MLP weights are missing");
        auto * original = layer.ffn_down;
        require(!ggml_is_quantized(original->type), "native TTT requires F16/F32 MLP weights");
        auto * weight = ggml_new_tensor_2d(state->ctx.get(), GGML_TYPE_F32, original->ne[0], original->ne[1]);
        state->down.push_back(weight);
        ggml_build_forward_expand(init, ggml_cpy(state->ctx.get(), original, weight));
    }
    auto * original_bias = model.layers.back().ffn_down_b;
    state->last_bias = ggml_new_tensor_1d(state->ctx.get(), GGML_TYPE_F32, original_bias->ne[0]);
    ggml_build_forward_expand(init, ggml_cpy(state->ctx.get(), original_bias, state->last_bias));
    if (state->constants_owner) {
        auto & constants = *state->constants_owner;
        state->embedding = constants.embedding;
        state->output = constants.output;
        state->norm = constants.norm;
        state->norm_bias = constants.norm_bias;
    } else {
        state->embedding = ggml_dup_tensor(state->ctx.get(), model.tok_embd);
        state->output = ggml_dup_tensor(state->ctx.get(), model.output);
        state->norm = ggml_dup_tensor(state->ctx.get(), model.output_norm);
        state->norm_bias = ggml_dup_tensor(state->ctx.get(), model.output_norm_b);
    }
    state->buffer.reset(ggml_backend_alloc_ctx_tensors(state->ctx.get(), state->backend.get()));
    require(bool(state->buffer), "cannot allocate native TTT weight buffer");
    require(ggml_backend_graph_compute(state->backend.get(), init) == GGML_STATUS_SUCCESS, "cannot initialize native TTT weights");
    // Embeddings may be CPU-resident even when all transformer blocks use CUDA.
    if (!state->constants_owner) {
        ggml_backend_tensor_copy(model.tok_embd, state->embedding);
        ggml_backend_tensor_copy(model.output, state->output);
        ggml_backend_tensor_copy(model.output_norm, state->norm);
        ggml_backend_tensor_copy(model.output_norm_b, state->norm_bias);
    }
    ggml_backend_sched_reset(ctx->sched.get());
    for (size_t i = 0; i < state->down.size(); ++i) model.layers[i].ffn_down = state->down[i];
    model.layers.back().ffn_down_b = state->last_bias;
    model.ttt_state = state;
    return *state;
}

void clear_cache(llama_context * ctx) {
    llama_kv_cache_clear(ctx);

}

int apply(llama_context * ctx, const char * text, float lr) {
    require(ctx != nullptr && std::isfinite(lr) && lr > 0 && lr <= 0.01f, "invalid native TTT arguments");
    auto & state = state_for(ctx, true);
    auto tokens = tokenize(ctx, text, true);
    require(tokens.size() >= 2, "TTT needs at least two tokens");
    capture(ctx, tokens, state.backend.get());
    work_graph work(state.backend.get());
    auto * c = work.ctx.get();
    auto * input = ggml_new_tensor_1d(c, GGML_TYPE_I32, tokens.size()-1);
    auto * embeddings = ggml_get_rows(c, state.embedding, input);
    auto * v = ggml_cont(c, ggml_transpose(c, embeddings));
    auto * graph = work.graph();
    std::vector<ggml_tensor *> candidates;
    std::vector<ggml_tensor *> norms;
    for (size_t i = 0; i < state.down.size(); ++i) {
        auto * captured = external(c, ctx->ttt_activations[i]);
        auto * z = ggml_view_2d(c, captured, captured->ne[0], tokens.size()-1, captured->nb[1], 0);
        auto * transposed = ggml_cont(c, ggml_transpose(c, z));
        auto * delta = ggml_scale(c, ggml_mul_mat(c, v, transposed), lr);
        // mul_mat(v, z) is [hidden, intermediate]; the stored weight is its transpose.
        auto * updated = ggml_add(c, state.down[i], ggml_cont(c, ggml_transpose(c, delta)));
        auto * norm = ggml_sum(c, ggml_sqr(c, updated));
        norms.push_back(norm);
        ggml_build_forward_expand(graph, norm);
        candidates.push_back(updated);
    }
    work.allocate();
    ggml_backend_tensor_set(input, tokens.data()+1, 0, (tokens.size()-1)*sizeof(llama_token));
    work.run(graph);
    require_norms(norms);
    auto * commit = work.graph();
    for (size_t i = 0; i < candidates.size(); ++i)
        ggml_build_forward_expand(commit, ggml_cpy(c, external(c, candidates[i]), state.down[i]));
    work.run(commit);
    clear_cache(ctx);
    return 0;
}

int learn_word(llama_context * ctx, const char * prefix, const char * output, int limit, int32_t * steps) {
    require(ctx && prefix && output && steps && limit > 0 && limit <= 128, "invalid word-learning arguments");
    *steps = 0;
    auto & state = state_for(ctx, true);
    const auto prefix_tokens = tokenize(ctx, prefix, false);
    auto tokens = tokenize(ctx, (std::string(prefix) + output).c_str(), true);
    require(tokens.size() > prefix_tokens.size(), "word target is empty");
    require(std::equal(prefix_tokens.begin(), prefix_tokens.end(), tokens.begin()), "word token boundary mismatch");
    capture(ctx, tokens, state.backend.get());
    const int n = int(tokens.size()-prefix_tokens.size());
    const int offset = int(prefix_tokens.size())-1;
    const int hidden = int(state.last_bias->ne[0]);
    const int vocab = int(ctx->model.tok_embd->ne[1]);
    work_graph work(state.backend.get());
    auto * c = work.ctx.get();
    auto * weight = state.down.back();
    auto * bias = state.last_bias;
    ggml_set_param(c, weight);
    ggml_set_param(c, bias);
    auto * source_z = external(c, ctx->ttt_activations.back());
    auto * z = ggml_view_2d(c, source_z, source_z->ne[0], n, source_z->nb[1], offset*source_z->nb[1]);
    auto * residual = ggml_view_2d(c, external(c, ctx->ttt_last_residual), hidden, n, ctx->ttt_last_residual->nb[1], offset*ctx->ttt_last_residual->nb[1]);
    auto * labels = ggml_new_tensor_2d(c, GGML_TYPE_F32, vocab, n);
    auto * one = scalar(c);
    auto * eps = scalar(c);
    auto * adam_params = ggml_new_tensor_1d(c, GGML_TYPE_F32, 7);
    auto * m_w = ggml_dup_tensor(c, weight);
    auto * v_w = ggml_dup_tensor(c, weight);
    auto * m_b = ggml_dup_tensor(c, bias);
    auto * v_b = ggml_dup_tensor(c, bias);
    auto * backup_w = ggml_dup_tensor(c, weight);
    auto * backup_b = ggml_dup_tensor(c, bias);
    auto * backup = work.graph();
    ggml_build_forward_expand(backup, ggml_cpy(c, weight, backup_w));
    ggml_build_forward_expand(backup, ggml_cpy(c, bias, backup_b));
    auto * rollback = work.graph();
    ggml_build_forward_expand(rollback, ggml_cpy(c, backup_w, weight));
    ggml_build_forward_expand(rollback, ggml_cpy(c, backup_b, bias));
    auto * x = ggml_add(c, residual, ggml_add(c, ggml_mul_mat(c, weight, z), bias));
    auto * normalized = ggml_norm(c, x, ctx->model.hparams.f_norm_eps);
    auto * final_hidden = ggml_add(c, ggml_mul(c, normalized, state.norm), state.norm_bias);
    auto * logits = ggml_mul_mat(c, state.output, final_hidden);
    auto * predictions = ggml_argmax(c, logits);
    auto * d_logits = ggml_cross_entropy_loss_back(c, one, logits, labels);
    auto * output_t = ggml_cont(c, ggml_transpose(c, state.output));
    auto * g = ggml_mul(c, ggml_mul_mat(c, output_t, d_logits), state.norm);
    auto * centered = ggml_sub(c, x, rows_mean(c, x));
    auto * stddev = ggml_sqrt(c, ggml_add(c, rows_mean(c, ggml_sqr(c, centered)), eps));
    auto * inv_std = ggml_div(c, ggml_repeat(c, one, stddev), stddev);
    auto * gradient = ggml_mul(c, ggml_sub(c, ggml_sub(c, g, rows_mean(c, g)),
        ggml_mul(c, normalized, rows_mean(c, ggml_mul(c, g, normalized)))), inv_std);
    auto * grad_w = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, z)), ggml_cont(c, ggml_transpose(c, gradient)));
    auto * grad_b = ggml_reshape_1d(c, ggml_sum_rows(c, ggml_cont(c, ggml_transpose(c, gradient))), hidden);
    auto * grad_norm = ggml_sqrt(c, ggml_add(c, ggml_sum(c, ggml_sqr(c, grad_w)), ggml_sum(c, ggml_sqr(c, grad_b))));
    auto * clip = ggml_div(c, one, ggml_clamp(c, grad_norm, 1.0f, std::numeric_limits<float>::max()));
    grad_w = ggml_mul(c, grad_w, clip);
    grad_b = ggml_mul(c, grad_b, clip);
    auto * forward = work.graph();
    ggml_build_forward_expand(forward, predictions);
    ggml_build_forward_expand(forward, grad_w);
    ggml_build_forward_expand(forward, grad_b);
    work.allocate();
    auto * optimize = work.graph();
    ggml_build_forward_expand(optimize, ggml_opt_step_adamw(c, weight, external(c, grad_w), m_w, v_w, adam_params));
    ggml_build_forward_expand(optimize, ggml_opt_step_adamw(c, bias, external(c, grad_b), m_b, v_b, adam_params));
    set_float(one, 1.0f);
    set_float(eps, ctx->model.hparams.f_norm_eps);
    std::vector<float> target(size_t(vocab)*n, 0.0f);
    for (int i = 0; i < n; ++i) target[size_t(i)*vocab+tokens[prefix_tokens.size()+i]] = 1.0f;
    ggml_backend_tensor_set(labels, target.data(), 0, target.size()*sizeof(float));
    work.run(backup);
    bool matched = false;
    const auto start = std::chrono::steady_clock::now();
    try {
        std::vector<int32_t> predicted(n);
        for (int iteration = 0; iteration <= limit; ++iteration) {
            work.run(forward);
            ggml_backend_tensor_get(predictions, predicted.data(), 0, predicted.size()*sizeof(int32_t));
            matched = std::equal(predicted.begin(), predicted.end(), tokens.begin()+prefix_tokens.size());
            if (matched || iteration == limit || std::chrono::steady_clock::now()-start > std::chrono::milliseconds(500)) break;
            const float beta1 = 0.9f, beta2 = 0.999f;
            float parameters[] = {0.002f, beta1, beta2, 1e-8f, 0.0f,
                1.0f/(1.0f-std::pow(beta1, float(iteration+1))), 1.0f/(1.0f-std::pow(beta2, float(iteration+1)))};
            ggml_backend_tensor_set(adam_params, parameters, 0, sizeof(parameters));
            work.run(optimize);
            *steps = iteration+1;
        }
        if (!matched) work.run(rollback);
    } catch (...) { work.run(rollback); clear_cache(ctx); throw; }
    clear_cache(ctx);
    return matched ? 1 : 0;
}

struct file_header {
    char magic[8];
    uint64_t cursor;
    uint32_t layers;
    uint32_t intermediate;
    uint32_t hidden;
    uint32_t reserved;
};

int save(llama_context * ctx, const char * path, uint64_t cursor) {
    require(ctx && path, "invalid snapshot arguments");
    auto & state = state_for(ctx);
    file_header header = {{'Z','T','T','T','0','0','0','1'}, cursor, uint32_t(state.down.size()),
        uint32_t(state.down[0]->ne[0]), uint32_t(state.down[0]->ne[1]), 0};
    const std::string temporary = std::string(path)+".tmp";
    const int fd = ::open(temporary.c_str(), O_WRONLY|O_CREAT|O_TRUNC, 0600);
    require(fd >= 0, "cannot create native TTT snapshot");
    auto write_all = [fd](const void * ptr, size_t size) {
        auto * bytes = static_cast<const char *>(ptr);
        while (size) { const auto written = ::write(fd, bytes, size); require(written > 0, "cannot write TTT snapshot"); bytes += written; size -= size_t(written); }
    };
    try {
        write_all(&header, sizeof(header));
        auto tensors = state.down;
        tensors.push_back(state.last_bias);
        for (auto * tensor : tensors) {
            std::vector<float> data(size_t(ggml_nelements(tensor)));
            ggml_backend_tensor_get(tensor, data.data(), 0, ggml_nbytes(tensor));
            double norm = 0;
            for (float value : data) { require(std::isfinite(value), "non-finite native TTT snapshot"); norm += double(value)*value; }
            require(norm > 0 || tensor == state.last_bias, "zero native TTT snapshot weight");
            write_all(data.data(), data.size()*sizeof(float));
        }
        require(::fsync(fd) == 0, "cannot sync native TTT snapshot");
        require(::close(fd) == 0, "cannot close native TTT snapshot");
        require(::rename(temporary.c_str(), path) == 0, "cannot publish native TTT snapshot");
        auto slash = std::string(path).find_last_of('/');
        const auto parent = slash == std::string::npos ? "." : std::string(path).substr(0, slash);
        const int directory = ::open(parent.c_str(), O_RDONLY|O_DIRECTORY);
        if (directory >= 0) { ::fsync(directory); ::close(directory); }
    } catch (...) { ::close(fd); ::unlink(temporary.c_str()); throw; }
    return 0;
}

int load(llama_context * ctx, const char * path, uint64_t * cursor) {
    require(ctx && path && cursor, "invalid snapshot arguments");
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream), "cannot open native TTT snapshot");
    file_header header{};
    stream.read(reinterpret_cast<char *>(&header), sizeof(header));
    require(bool(stream) && std::memcmp(header.magic, "ZTTT0001", 8) == 0, "invalid native TTT snapshot header");
    const auto & model = ctx->model;
    require(header.layers == model.hparams.n_layer && header.hidden == model.hparams.n_embd &&
        header.intermediate == model.layers[0].ffn_down->ne[0], "native TTT snapshot shape mismatch");
    // Validate all bytes before changing live weights.
    std::vector<std::vector<float>> data;
    for (uint32_t i = 0; i <= header.layers; ++i) {
        const size_t count = i == header.layers ? header.hidden : size_t(header.intermediate)*header.hidden;
        data.emplace_back(count);
        stream.read(reinterpret_cast<char *>(data.back().data()), count*sizeof(float));
        require(bool(stream), "incomplete native TTT snapshot");
        double norm = 0;
        for (float value : data.back()) { require(std::isfinite(value), "invalid native TTT snapshot value"); norm += double(value)*value; }
        require(norm > 0 || i == header.layers, "zero native TTT snapshot weight");
    }
    require(stream.peek() == std::char_traits<char>::eof(), "unexpected native TTT snapshot bytes");
    auto & state = state_for(ctx, true);
    for (size_t i = 0; i < state.down.size(); ++i)
        ggml_backend_tensor_set(state.down[i], data[i].data(), 0, data[i].size()*sizeof(float));
    ggml_backend_tensor_set(state.last_bias, data.back().data(), 0, data.back().size()*sizeof(float));
    *cursor = header.cursor;
    clear_cache(ctx);
    return 0;
}
}

namespace {
std::mutex retirement_mutex;
std::vector<std::shared_ptr<llama_ttt_state>> retired_states;
}

struct llama_ttt_version {
    std::shared_ptr<llama_ttt_state> state;
};

extern "C" {
llama_ttt_version * llama_ttt_export_version(llama_context * ctx) {
    try {
        last_error.clear();
        // Release old GPU buffers on the background training thread.
        std::vector<std::shared_ptr<llama_ttt_state>> release;
        { std::lock_guard<std::mutex> lock(retirement_mutex); release.swap(retired_states); }
        require(ctx != nullptr, "missing native TTT context");
        state_for(ctx);
        return new llama_ttt_version{ctx->model.ttt_state};
    } catch (const std::exception & error) { last_error = error.what(); return nullptr; }
}
void llama_ttt_free_version(llama_ttt_version * version) { delete version; }
int32_t llama_ttt_import_version(llama_context * ctx, const llama_ttt_version * version) {
    try {
        last_error.clear();
        require(ctx && version && version->state, "missing native TTT version");
        auto & model = const_cast<llama_model &>(ctx->model);
        auto & state = *version->state;
        require(model.arch == LLM_ARCH_GPT2 && model.layers.size() == state.down.size(), "native TTT version architecture mismatch");
        for (size_t i = 0; i < state.down.size(); ++i) {
            auto * weight = model.layers[i].ffn_down;
            require(ggml_are_same_shape(weight, state.down[i]), "native TTT version shape mismatch");
            auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(weight->buffer));
            require(device == ggml_backend_get_device(state.backend.get()), "native TTT version device mismatch");
        }
        llama_synchronize(ctx);
        ggml_backend_sched_reset(ctx->sched.get());
        for (size_t i = 0; i < state.down.size(); ++i) model.layers[i].ffn_down = state.down[i];
        model.layers.back().ffn_down_b = state.last_bias;
        auto previous = model.ttt_state;
        model.ttt_state = version->state;
        if (previous && previous != version->state) {
            std::lock_guard<std::mutex> lock(retirement_mutex);
            retired_states.push_back(std::move(previous));
        }
        clear_cache(ctx);
        return 0;
    } catch (const std::exception & error) { last_error = error.what(); return -1; }
}
const char * llama_ttt_last_error(void) { return last_error.c_str(); }
int32_t llama_ttt_apply(llama_context * ctx, const char * text, float lr) {
    try { last_error.clear(); return apply(ctx, text, lr); }
    catch (const std::exception & error) { last_error = error.what(); if (ctx) clear_cache(ctx); return -1; }
}
int32_t llama_ttt_learn_word(llama_context * ctx, const char * prefix, const char * output, int32_t limit, int32_t * steps) {
    try { last_error.clear(); return learn_word(ctx, prefix, output, limit, steps); }
    catch (const std::exception & error) { last_error = error.what(); if (ctx) clear_cache(ctx); return -1; }
}
int32_t llama_ttt_save(llama_context * ctx, const char * path, uint64_t cursor) {
    try { last_error.clear(); return save(ctx, path, cursor); }
    catch (const std::exception & error) { last_error = error.what(); return -1; }
}
int32_t llama_ttt_load(llama_context * ctx, const char * path, uint64_t * cursor) {
    try { last_error.clear(); return load(ctx, path, cursor); }
    catch (const std::exception & error) { last_error = error.what(); return -1; }
}
}
