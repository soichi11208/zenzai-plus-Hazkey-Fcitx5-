// SPDX-License-Identifier: Unlicense
#include "llama.h"
#include "ggml-backend.h"
#include "llama-context.h"
#include "llama-model.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using clock_type = std::chrono::steady_clock;
static double milliseconds(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now()-start).count();
}
static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
static void api(int result) {
    if (result < 0) throw std::runtime_error(llama_ttt_last_error());
}
static std::vector<llama_token> tokens(llama_context * ctx, const std::string & text, bool eos = false) {
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, false, true);
    std::vector<llama_token> result(n);
    check(llama_tokenize(vocab, text.c_str(), text.size(), result.data(), n, false, true) == n, "tokenize failed");
    if (eos) result.push_back(llama_vocab_eos(vocab));
    return result;
}
static void decode(llama_context * ctx, const std::vector<llama_token> & ids, bool all = true) {
    llama_kv_cache_clear(ctx);
    auto batch = llama_batch_init(ids.size(), 0, 1);
    batch.n_tokens = ids.size();
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = ids[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
        batch.logits[i] = all || i+1 == batch.n_tokens;
    }
    const int result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    check(result == 0, "decode failed");
    llama_synchronize(ctx);
}
static double nll(llama_context * ctx, const std::string & prefix, const std::string & output) {
    auto full = tokens(ctx, prefix+output, true);
    auto pre = tokens(ctx, prefix);
    decode(ctx, full);
    const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    double loss = 0;
    for (size_t i = pre.size(); i < full.size(); ++i) {
        const auto * logits = llama_get_logits_ith(ctx, int(i)-1);
        const float maximum = *std::max_element(logits, logits+vocabulary);
        double total = 0;
        for (int k = 0; k < vocabulary; ++k) {
            check(std::isfinite(logits[k]), "non-finite logits");
            total += std::exp(double(logits[k])-maximum);
        }
        loss += maximum+std::log(total)-logits[full[i]];
    }
    return loss/double(full.size()-pre.size());
}
static std::string generate(llama_context * ctx, const std::string & prefix) {
    auto ids = tokens(ctx, prefix);
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int count = llama_vocab_n_tokens(vocab);
    std::string result;
    for (int i = 0; i < 40; ++i) {
        decode(ctx, ids, false);
        const auto * logits = llama_get_logits_ith(ctx, -1);
        auto next = llama_token(std::max_element(logits, logits+count)-logits);
        if (llama_vocab_is_eog(vocab, next)) return result;
        char piece[256];
        int size = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, true);
        check(size >= 0, "token piece failed");
        result.append(piece, size); ids.push_back(next);
    }
    return result;
}
static std::vector<float> weight(llama_context * ctx, int layer) {
    auto * tensor = ctx->model.layers.at(layer).ffn_down;
    const int64_t n = ggml_nelements(tensor);
    std::vector<float> result(n);
    if (tensor->type == GGML_TYPE_F32) ggml_backend_tensor_get(tensor, result.data(), 0, n*sizeof(float));
    else {
        check(tensor->type == GGML_TYPE_F16, "unexpected weight type");
        std::vector<ggml_fp16_t> half(n);
        ggml_backend_tensor_get(tensor, half.data(), 0, n*sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(half.data(), result.data(), n);
    }
    return result;
}
static void dump(const std::string & path, const std::vector<float> & data) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data.data()), data.size()*sizeof(float));
    check(bool(f), "dump failed");
}
static double difference(const std::vector<float> & a, const std::vector<float> & b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) { check(std::isfinite(b[i]), "non-finite weight"); sum += std::pow(double(a[i])-b[i], 2); }
    return std::sqrt(sum);
}
struct instance {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    instance(const char * path, ggml_backend_dev_t device) {
        ggml_backend_dev_t devices[] = {device, nullptr};
        auto params = llama_model_default_params(); params.devices = devices; params.n_gpu_layers = 99; params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model = llama_model_load_from_file(path, params); check(model, "model load failed");
        auto cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
        cp.n_threads = 4; cp.n_threads_batch = 4;
        ctx = llama_init_from_model(model, cp); check(ctx, "context load failed");
    }
    ~instance() { if (ctx) llama_free(ctx); if (model) llama_model_free(model); }
};
int main(int argc, char ** argv) {
    if (argc < 4) return 2;
    try {
        std::string backend_path = argv[1]; if (backend_path.back() != '/') backend_path += '/';
        ggml_backend_load_all_from_path(backend_path.c_str()); llama_backend_init();
        auto device = ggml_backend_dev_by_name("CUDA1"); check(device, "CUDA1 unavailable");
        printf("DEVICE %s\n", ggml_backend_dev_description(device)); fflush(stdout);
        instance trainer(argv[2], device);
        const std::string prefix = "\uEE00ムソウカキョウ\uEE01";
        const std::string output = "夢想夏郷";
        const std::string directory = argv[3];
        auto promoted = llama_ttt_export_version(trainer.ctx); check(promoted, "promotion failed");
        llama_ttt_free_version(promoted);
        auto before = weight(trainer.ctx, 0);
        const double before_loss = nll(trainer.ctx, prefix, output);
        printf("BEFORE nll=%.9f generated=%s\n", before_loss, generate(trainer.ctx, prefix).c_str()); fflush(stdout);
        auto ids = tokens(trainer.ctx, prefix+output, true);
        for (int layer = 0; layer < int(trainer.ctx->model.hparams.n_layer); ++layer)
            dump(directory+"/before-"+std::to_string(layer)+".bin", weight(trainer.ctx, layer));
        auto start = clock_type::now(); api(llama_ttt_apply(trainer.ctx, (prefix+output).c_str(), 1e-4f));
        const double first_ms = milliseconds(start);
        for (int layer = 0; layer < int(trainer.ctx->model.hparams.n_layer); ++layer) {
            auto * z = trainer.ctx->ttt_activations.at(layer); check(z, "reference activation missing");
            std::vector<float> data(ggml_nelements(z)); ggml_backend_tensor_get(z, data.data(), 0, data.size()*4);
            dump(directory+"/z-"+std::to_string(layer)+".bin", data);
        }
        auto * emb = trainer.ctx->model.tok_embd;
        std::vector<ggml_fp16_t> half(ggml_nelements(emb));
        check(emb->type == GGML_TYPE_F16, "expected f16 reference embeddings");
        ggml_backend_tensor_get(emb, half.data(), 0, half.size()*2);
        std::vector<float> embeddings(half.size()); ggml_fp16_to_fp32_row(half.data(), embeddings.data(), half.size());
        dump(directory+"/embedding.bin", embeddings);
        { std::ofstream f(directory+"/tokens.txt"); for (auto id : ids) f << id << "\n"; }
        auto after = weight(trainer.ctx, 0);
        for (int layer = 0; layer < int(trainer.ctx->model.hparams.n_layer); ++layer)
            dump(directory+"/after-"+std::to_string(layer)+".bin", weight(trainer.ctx, layer));
        printf("TTT first_ms=%.6f delta_norm=%.9f nll_before=%.9f nll_after=%.9f\n", first_ms, difference(before,after), before_loss, nll(trainer.ctx,prefix,output)); fflush(stdout);
        start = clock_type::now(); int32_t steps = 0;
        const int matched = llama_ttt_learn_word(trainer.ctx, prefix.c_str(), output.c_str(), 32, &steps); api(matched);
        const double word_ms = milliseconds(start);
        printf("WORD matched=%d steps=%d ms=%.6f nll=%.9f generated=%s\n", matched, steps, word_ms, nll(trainer.ctx,prefix,output), generate(trainer.ctx,prefix).c_str()); fflush(stdout);
        check(matched == 1 && generate(trainer.ctx,prefix)==output, "word learning did not converge");
        const auto snapshot = directory+"/snapshot.bin";
        start = clock_type::now(); api(llama_ttt_save(trainer.ctx,snapshot.c_str(),42));
        printf("SAVE ms=%.6f\n", milliseconds(start)); fflush(stdout);
        instance inference(argv[2],device);
        const auto inference_before = weight(inference.ctx,0);
        auto version = llama_ttt_export_version(trainer.ctx); check(version, "export version failed");
        start = clock_type::now(); api(llama_ttt_import_version(inference.ctx,version));
        const double publish_ms = milliseconds(start);
        printf("PUBLISH ms=%.6f generated=%s\n",publish_ms,generate(inference.ctx,prefix).c_str()); fflush(stdout);
        check(generate(inference.ctx,prefix)==output,"publication failed");
        const auto frozen = weight(inference.ctx,0);
        start = clock_type::now(); api(llama_ttt_apply(trainer.ctx,"\uEE00チャントウゴイテル\uEE01ちゃんと動いてる",1e-4f));
        const double cow_ms = milliseconds(start);
        printf("COW ms=%.6f inference_delta=%.9f trainer_delta=%.9f\n",cow_ms,difference(frozen,weight(inference.ctx,0)),difference(frozen,weight(trainer.ctx,0))); fflush(stdout);
        check(difference(frozen,weight(inference.ctx,0))==0,"published version was mutated");
        uint64_t cursor = 0; start=clock_type::now(); api(llama_ttt_load(trainer.ctx,snapshot.c_str(),&cursor));
        const double load_ms = milliseconds(start);
        printf("LOAD ms=%.6f cursor=%llu generated=%s\n",load_ms,(unsigned long long)cursor,generate(trainer.ctx,prefix).c_str()); fflush(stdout);
        check(cursor==42 && generate(trainer.ctx,prefix)==output,"snapshot restore failed");
        llama_ttt_free_version(version);
        const auto intact = weight(trainer.ctx,0);
        { std::ifstream source(snapshot,std::ios::binary);std::ofstream broken(directory+"/broken.bin",std::ios::binary);char header[100];source.read(header,sizeof(header));broken.write(header,source.gcount()); }
        const int invalid_snapshot=llama_ttt_load(trainer.ctx,(directory+"/broken.bin").c_str(),&cursor);
        check(invalid_snapshot<0 && difference(intact,weight(trainer.ctx,0))==0,"invalid snapshot changed live weights");
        check(llama_ttt_apply(trainer.ctx,"",1e-4f)<0 && difference(intact,weight(trainer.ctx,0))==0,"invalid input changed live weights");
        printf("INVALID_INPUT_AND_SNAPSHOT rejected_without_weight_changes\n");
        std::vector<double> normal_ms;
        for(int i=0;i<8;++i){start=clock_type::now();api(llama_ttt_apply(trainer.ctx,(prefix+output).c_str(),1e-4f));normal_ms.push_back(milliseconds(start));}
        std::sort(normal_ms.begin(),normal_ms.end());
        printf("STEADY_TTT p50_ms=%.6f max_ms=%.6f\n",normal_ms[normal_ms.size()/2],normal_ms.back()); fflush(stdout);
        // A fresh baseline for independent quality checks.
        instance quality(argv[2], device);
        auto ready = llama_ttt_export_version(quality.ctx); check(ready, "quality model promotion failed");
        llama_ttt_free_version(ready);
        const auto clean = directory+"/clean.bin";
        api(llama_ttt_save(quality.ctx,clean.c_str(),0));
        const std::pair<std::string,std::string> cases[] = {
            {"ムソウカキョウ", "夢想夏郷"}, {"ショクモツセンイ", "食物繊維"},
            {"サイキドウ", "再起動"}, {"セイシン", "精神"},
            {"チャントウゴイテル", "ちゃんと動いてる"}
        };
        for (const auto & row : cases) {
            api(llama_ttt_load(quality.ctx,clean.c_str(),&cursor));
            const std::string prompt = "\uEE00"+row.first+"\uEE01";
            const double original = nll(quality.ctx,prompt,row.second);
            start=clock_type::now(); api(llama_ttt_apply(quality.ctx,(prompt+row.second).c_str(),1e-4f));
            const double update_ms=milliseconds(start);
            printf("QUALITY reading=%s before=%.9f after=%.9f ms=%.6f generated=%s\n",row.first.c_str(),original,nll(quality.ctx,prompt,row.second),update_ms,generate(quality.ctx,prompt).c_str()); fflush(stdout);
        }
        for (const auto & row : std::vector<std::pair<std::string,std::string>>{
                {"ムソウカキョウ", "夢想夏郷"}, {"コーデックスネコ", "Codex猫"}, {"クウソウカイロウ", "空想回廊"}}) {
            api(llama_ttt_load(quality.ctx,clean.c_str(),&cursor));
            const std::string prompt="\uEE00"+row.first+"\uEE01";
            int32_t count=0; start=clock_type::now(); const int ok=llama_ttt_learn_word(quality.ctx,prompt.c_str(),row.second.c_str(),32,&count); api(ok);
            const double learned_ms=milliseconds(start);
            const auto text=generate(quality.ctx,prompt);
            printf("WORD_CASE target=%s matched=%d steps=%d ms=%.6f generated=%s\n",row.second.c_str(),ok,count,learned_ms,text.c_str()); fflush(stdout);
            check(ok==1 && text==row.second,"word case failed");
        }
        api(llama_ttt_load(quality.ctx,clean.c_str(),&cursor));
        const std::vector<std::pair<std::string,std::string>> registered={
            {"ムソウカキョウ", "夢想夏郷"}, {"コーデックスネコ", "Codex猫"}, {"クウソウカイロウ", "空想回廊"}};
        for(const auto & row:registered){int32_t count=0;const auto prompt="\uEE00"+row.first+"\uEE01";api(llama_ttt_learn_word(quality.ctx,prompt.c_str(),row.second.c_str(),32,&count));}
        for(const auto & row:registered){const auto text=generate(quality.ctx,"\uEE00"+row.first+"\uEE01");printf("RETENTION target=%s generated=%s matched=%d\n",row.second.c_str(),text.c_str(),int(text==row.second));fflush(stdout);}
        const auto timing_ids=tokens(inference.ctx,"\uEE00チャントウゴイテル\uEE01");
        auto inference_times=[&](){
            std::vector<double> samples;
            for(int i=0;i<40;++i){auto begin=clock_type::now();decode(inference.ctx,timing_ids,false);samples.push_back(milliseconds(begin));std::this_thread::sleep_for(std::chrono::milliseconds(10));}
            std::sort(samples.begin(),samples.end());return samples;
        };
        auto baseline=inference_times();
        std::atomic<bool> go{false};
        std::exception_ptr background_error;
        std::vector<double> background_times;
        std::thread background([&]{
            try {
                while(!go.load())std::this_thread::yield();
                llama_ttt_version * held=nullptr;
                for(int i=0;i<8;++i){
                    auto begin=clock_type::now();api(llama_ttt_apply(trainer.ctx,(prefix+output).c_str(),1e-4f));
                    api(llama_ttt_save(trainer.ctx,(directory+"/concurrent.bin").c_str(),43+i));
                    auto * next=llama_ttt_export_version(trainer.ctx);check(next,"concurrent export failed");
                    if(held)llama_ttt_free_version(held);held=next;
                    background_times.push_back(milliseconds(begin));
                }
                if(held)llama_ttt_free_version(held);
            } catch(...) {background_error=std::current_exception();}
        });
        go=true;auto concurrent=inference_times();background.join();
        if(background_error)std::rethrow_exception(background_error);
        printf("INFERENCE_BASELINE p50_ms=%.6f p95_ms=%.6f max_ms=%.6f\n",baseline[20],baseline[38],baseline.back());
        printf("INFERENCE_CONCURRENT p50_ms=%.6f p95_ms=%.6f max_ms=%.6f\n",concurrent[20],concurrent[38],concurrent.back());
        std::sort(background_times.begin(),background_times.end());
        printf("CONCURRENT_TRAIN_SAVE p50_ms=%.6f max_ms=%.6f\n",background_times[4],background_times.back());
        check(difference(frozen,weight(inference.ctx,0))==0,"concurrent training changed inference weights");
        printf("CORE_CHECKS_PASS (quality and retention diagnostics reported separately)\n"); return 0;
    } catch(const std::exception & e){fprintf(stderr,"PROBE_FAILURE %s\n",e.what());return 1;}
}
