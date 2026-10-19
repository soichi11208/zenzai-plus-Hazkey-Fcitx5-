#if Zenzai
// Zenzai が有効でない場合、llama-mock.swift の実装が利用される
import llama
#endif

import Algorithms
import EfficientNGram
import Foundation
import HeapModule
import SwiftUtils

public typealias ZenzaiDeviceConfig = ConvertRequestOptions.ZenzaiMode.DeviceConfig

public struct GGMLBackendDevice: Sendable {
    public let name: String
    public let description: String
    public let type: DeviceType

    public enum DeviceType: Sendable {
        case cpu
        case gpu
        case accel
        case unknown
    }

    #if Zenzai
    init(device: ggml_backend_dev_t) {
        if let namePtr = ggml_backend_dev_name(device) {
            self.name = String(cString: namePtr)
        } else {
            self.name = "Unknown"
        }

        if let descPtr = ggml_backend_dev_description(device) {
            self.description = String(cString: descPtr)
        } else {
            self.description = "Unknown"
        }

        let deviceType = ggml_backend_dev_type(device)
        switch deviceType {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            self.type = .cpu
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            self.type = .gpu
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            self.type = .accel
        default:
            self.type = .unknown
        }
    }
    #endif
}

/// Enumerate available GGML backend devices
/// Note: loadGGMLBackends() should be called once before using this function
public func enumerateGGMLBackendDevices() -> [GGMLBackendDevice] {
    #if Zenzai
    let deviceCount = ggml_backend_dev_count()
    var devices: [GGMLBackendDevice] = []
    for i in 0..<deviceCount {
        if let device = ggml_backend_dev_get(i) {
            devices.append(GGMLBackendDevice(device: device))
        }
    }
    return devices
    #else
    return []
    #endif
}

/// Load all available GGML backends
/// - Parameter path: Optional directory path to load backends from. If nil, uses default search paths.
/// This function should be called once at application startup before using any Zenzai features
public func loadGGMLBackends(from path: String? = nil) {
    #if Zenzai
    if let path = path {
        ggml_backend_load_all_from_path(path)
    } else {
        ggml_backend_load_all()
    }
    #endif
}

/// Create a device configuration based on device type detection
/// - Parameters:
///   - deviceName: Optional device name. If nil, uses the best available device.
///   - gpuLayers: Number of GPU layers when GPU is used. Default is 99.
/// - Returns: A configured ZenzaiDeviceConfig
public func createDeviceConfig(deviceName: String? = nil, gpuLayers: Int32 = 99) -> ZenzaiDeviceConfig {
    #if Zenzai
    let devices = enumerateGGMLBackendDevices()

    // If a specific device name is provided, try to find it
    if let targetName = deviceName {
        if let device = devices.first(where: { $0.name == targetName }) {
            switch device.type {
            case .gpu:
                return ZenzaiDeviceConfig(deviceName: targetName, gpuLayers: gpuLayers)
            case .cpu, .accel, .unknown:
                return ZenzaiDeviceConfig(deviceName: targetName, gpuLayers: 0)
            }
        }
    }

    // Fall back to CPU
    if let cpuDevice = devices.first(where: { $0.type == .cpu }) {
        return ZenzaiDeviceConfig(deviceName: cpuDevice.name, gpuLayers: 0)
    }

    // Default configuration
    return ZenzaiDeviceConfig(deviceName: nil, gpuLayers: 0)
    #else
    return ZenzaiDeviceConfig(deviceName: nil, gpuLayers: 0)
    #endif
}

struct FixedSizeHeap<Element: Comparable> {
    private var size: Int
    private var heap: Heap<Element>

    init(size: Int) {
        self.size = size
        self.heap = []
    }

    mutating func removeMax() {
        self.heap.removeMax()
    }

    mutating func removeMin() {
        self.heap.removeMin()
    }

    @discardableResult
    mutating func insertIfPossible(_ element: Element) -> Bool {
        if self.heap.count < self.size {
            self.heap.insert(element)
            return true
        } else if let min = self.heap.min, element > min {
            self.heap.replaceMin(with: element)
            return true
        } else {
            return false
        }
    }

    var unordered: [Element] {
        self.heap.unordered
    }

    var max: Element? {
        self.heap.max
    }

    var min: Element? {
        self.heap.min
    }

    var isEmpty: Bool {
        self.heap.isEmpty
    }
}

enum ZenzError: LocalizedError {
    case couldNotLoadModel(path: String)
    case couldNotLoadContext
    case couldNotLoadVocab
    case noDevicesAvailable

    var errorDescription: String? {
        switch self {
        case .couldNotLoadContext: return "failed to load context"
        case .couldNotLoadModel(path: let path): return "could not load model weight at \(path)"
        case .couldNotLoadVocab: return "failed to load vocab"
        case .noDevicesAvailable: return "no llama.cpp compatible device available"
        }
    }
}

final class ZenzContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var prevInput: [llama_token] = []
    private var prevPrompt: [llama_token] = []
    private var currentDeviceConfig: ZenzaiDeviceConfig
    private var modelPath: String

    private let n_len: Int32 = 512

    init(model: OpaquePointer, context: OpaquePointer, vocab: OpaquePointer, deviceConfig: ZenzaiDeviceConfig, modelPath: String) {
        self.model = model
        self.context = context
        self.vocab = vocab
        self.currentDeviceConfig = deviceConfig
        self.modelPath = modelPath
    }

    deinit {
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }

    private static func ctx_params(deviceConfig: ZenzaiDeviceConfig, contextLength: Int32 = 512) -> llama_context_params {
        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        debug("Using \(n_threads) threads")
        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = UInt32(contextLength)
        ctx_params.n_threads       = Int32(n_threads)
        ctx_params.n_threads_batch = Int32(n_threads)
        ctx_params.n_batch = UInt32(contextLength)
        ctx_params.n_ubatch = UInt32(contextLength)

        // Configure offload_kqv based on device type
        #if Zenzai
        if deviceConfig.gpuLayers > 0 {
            ctx_params.offload_kqv = true
        } else {
            ctx_params.offload_kqv = false
        }
        #endif

        return ctx_params
    }

    static func createContext(path: String, deviceConfig: ZenzaiDeviceConfig = ZenzaiDeviceConfig(), contextLength: Int32 = 512) throws -> ZenzContext {
        llama_backend_init()
        ggml_backend_load_all();

        let deviceCount = ggml_backend_dev_count()
        if deviceCount == 0 {
            debug("No backend devices available")
            throw ZenzError.noDevicesAvailable
        }

        var model_params = llama_model_default_params()
        model_params.use_mmap = true

        #if Zenzai
        // Configure GPU layers based on device config
        model_params.n_gpu_layers = deviceConfig.gpuLayers

        // Set device if specified
        if let deviceName = deviceConfig.deviceName {
            if let device = ggml_backend_dev_by_name(deviceName) {
                var deviceArray = [device, nil]
                let devicePtr = UnsafeMutablePointer<ggml_backend_dev_t?>.allocate(capacity: 2)
                devicePtr.initialize(from: &deviceArray, count: 2)
                model_params.devices = devicePtr
            }
        }
        #endif

        let model = llama_model_load_from_file(path, model_params)

        #if Zenzai
        // Free the allocated device pointer if it was set
        if model_params.devices != nil {
            model_params.devices?.deallocate()
        }
        #endif

        guard let model else {
            debug("Could not load model at \(path)")
            throw ZenzError.couldNotLoadModel(path: path)
        }

        let params = ctx_params(deviceConfig: deviceConfig, contextLength: contextLength)
        let context = llama_init_from_model(model, params)
        guard let context else {
            debug("Could not load context!")
            throw ZenzError.couldNotLoadContext
        }

        let vocab = llama_model_get_vocab(model)
        guard let vocab else {
            debug("Could not load vocab!")
            throw ZenzError.couldNotLoadVocab
        }

        return ZenzContext(model: model, context: context, vocab: vocab, deviceConfig: deviceConfig, modelPath: path)
    }

    /// Update device configuration dynamically
    /// This will unload and reload the model with new parameters
    func updateDeviceConfig(_ newConfig: ZenzaiDeviceConfig) throws {
        // Store the new config
        self.currentDeviceConfig = newConfig

        // Free existing resources
        llama_free(self.context)
        llama_model_free(self.model)

        // Reload model with new parameters
        var model_params = llama_model_default_params()
        model_params.use_mmap = true

        #if Zenzai
        // Configure GPU layers based on device config
        model_params.n_gpu_layers = newConfig.gpuLayers

        // Set device if specified
        if let deviceName = newConfig.deviceName {
            if let device = ggml_backend_dev_by_name(deviceName) {
                var deviceArray = [device, nil]
                let devicePtr = UnsafeMutablePointer<ggml_backend_dev_t?>.allocate(capacity: 2)
                devicePtr.initialize(from: &deviceArray, count: 2)
                model_params.devices = devicePtr
            }
        }
        #endif

        let model = llama_model_load_from_file(self.modelPath, model_params)

        #if Zenzai
        // Free the allocated device pointer if it was set
        if model_params.devices != nil {
            model_params.devices?.deallocate()
        }
        #endif

        guard let model else {
            debug("Could not reload model at \(self.modelPath)")
            throw ZenzError.couldNotLoadModel(path: self.modelPath)
        }
        self.model = model

        // Reload context with new parameters
        let params = Self.ctx_params(deviceConfig: newConfig)
        let context = llama_init_from_model(model, params)
        guard let context else {
            debug("Could not load context!")
            throw ZenzError.couldNotLoadContext
        }
        self.context = context

        // Reload vocab
        let vocab = llama_model_get_vocab(model)
        guard let vocab else {
            debug("Could not load vocab!")
            throw ZenzError.couldNotLoadVocab
        }
        self.vocab = vocab

        // Reset state
        self.prevInput = []
        self.prevPrompt = []
    }

    /// Get current device configuration
    func getDeviceConfig() -> ZenzaiDeviceConfig {
        return self.currentDeviceConfig
    }

    func reset_context() throws {
        llama_free(self.context)
        let params = Self.ctx_params(deviceConfig: self.currentDeviceConfig)
        let context = llama_init_from_model(self.model, params)
        guard let context else {
            debug("Could not load context!")
            throw ZenzError.couldNotLoadContext
        }
        self.context = context
        self.prevInput = []
        self.prevPrompt = []
    }

    private func get_logits(tokens: [llama_token], logits_start_index: Int = 0) -> UnsafeMutablePointer<Float>? {
        // Manage KV cache: remove entries that differ from previous input
        let prefixCacheCount: Int
        do {
            let pos_max = llama_kv_cache_seq_pos_max(self.context, 0)
            debug("pos max:", pos_max, "prevInput count:", self.prevInput.count, "tokens count:", tokens.count)
            let commonTokens = self.prevInput.commonPrefix(with: tokens)
            // Remove KV cache from position commonTokens.count onwards to recompute divergent part
            // removed range: [llama_pos(commonTokens.count), inf)
            prefixCacheCount = min(commonTokens.count, logits_start_index)
            llama_kv_cache_seq_rm(context, -1, llama_pos(prefixCacheCount), -1)
            debug("new pos max:", llama_kv_cache_seq_pos_max(self.context, 0), "commonTokens:", commonTokens.count)
        }
        var batch = llama_batch_init(512, 0, 1)
        defer { llama_batch_free(batch) }
        let n_ctx = llama_n_ctx(context)
        let n_kv_req = tokens.count + (Int(n_len) - tokens.count)
        if n_kv_req > n_ctx {
            debug("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
        }
        for i in tokens.indices.dropFirst(prefixCacheCount) {
            llama_batch_add(&batch, tokens[i], Int32(i), [0], logits: logits_start_index <= i)
        }
        // 評価
        if llama_decode(context, batch) != 0 {
            debug("llama_decode() failed")
            return nil
        }
        // update cached input for next call (for KV cache management)
        self.prevInput = tokens
        return llama_get_logits(context)
    }

    func evaluate(text: String, ignorePrompt: String = "") -> Float {
        let tokens_list = self.tokenize(text: text, add_bos: true, add_eos: true)
        guard let logits = self.get_logits(tokens: tokens_list) else {
            debug("logits unavailable")
            return .nan
        }
        let tokenizedPromptCount = ignorePrompt.isEmpty ? 1 : tokenize(text: ignorePrompt, add_bos: true, add_eos: false).count
        let n_vocab = llama_vocab_n_tokens(vocab)

        var sum: Float = 0
        // 最初のプロンプト部分は無視する
        for (i, token_id) in tokens_list.indexed().dropFirst(tokenizedPromptCount) {
            // FIXME: there can be more efficient implementations, poossibly using Accelerate or other frameworks.
            var log_prob: Float = 0
            for index in ((i - 1) * Int(n_vocab)) ..< (i * Int(n_vocab)) {
                log_prob += expf(logits[index])
            }
            log_prob = logf(log_prob)
            log_prob = logits[Int((i - 1) * Int(n_vocab) + Int(token_id))] - log_prob
            sum += log_prob
        }
        return sum
    }

    enum CandidateEvaluationResult: Sendable, Equatable, Hashable {
        case error
        case pass(score: Float, alternativeConstraints: [AlternativeConstraint])
        case fixRequired(prefixConstraint: [UInt8])
        case wholeResult(String)

        struct AlternativeConstraint: Sendable, Equatable, Hashable {
            var probabilityRatio: Float
            var prefixConstraint: [UInt8]
        }
    }

    func getLearningPriority(data: DicdataElement) -> Float {
        // 文字数の長い候補ほど優先的に適用されるようにする
        // 積極的な複合語化の効果を期待
        if 1 <= data.ruby.count && data.ruby.count <= 4 {
            Float(data.ruby.count + 2)
        } else if 5 <= data.ruby.count && data.ruby.count <= 15 {
            Float(data.ruby.count * 2)
        } else {
            30
        }
    }

    /// ピュアな貪欲法による生成を行って返す
    func pure_greedy_decoding(leftSideContext: String, maxCount: Int = .max) -> String {
        var prompt_tokens = self.tokenize(text: leftSideContext, add_bos: false)
        let initial_count = prompt_tokens.count
        let eos_token = llama_vocab_eos(vocab)
        while prompt_tokens.count - initial_count < maxCount {
            let startOffset = prompt_tokens.count - 1
            guard let logits = self.get_logits(tokens: prompt_tokens, logits_start_index: startOffset) else {
                debug("logits unavailable")
                return ""
            }
            let n_vocab = llama_vocab_n_tokens(vocab)
            let startIndex = (prompt_tokens.count - 1 - startOffset) * Int(n_vocab)
            let endIndex = (prompt_tokens.count - startOffset) * Int(n_vocab)
            // Min-Heapを使用してn-bestを計算
            var max_token: llama_token = -1
            var max_value: Float = Float.infinity * -1
            for index in startIndex..<endIndex {
                let token = llama_token(index - startIndex)
                if max_value < logits[index] {
                    max_token = token
                    max_value = logits[index]
                }
            }
            if max_token == eos_token {
                break
            } else {
                prompt_tokens.append(max_token)
            }
        }

        // Heapからソートして結果を取り出す
        let cchars: [CChar] = prompt_tokens.dropFirst(initial_count).flatMap(self.token_to_piece)
        let data = Data(cchars.map { UInt8(bitPattern: $0) })
        return String(data: data, encoding: .utf8) ?? "" as String
    }

    func predict_next_character(leftSideContext: String, count: Int) -> [(character: Character, value: Float)] {
        struct NextCharacterCandidate: Comparable {
            static func < (lhs: NextCharacterCandidate, rhs: NextCharacterCandidate) -> Bool {
                lhs.value < rhs.value
            }
            var character: Character
            var value: Float
        }

        // 文末を目指して生成するためのプロンプト
        // \u{EE01}を停止トークンとみなせる
        let prompt_tokens = self.tokenize(text: "\u{EE00}。\u{EE02}\(leftSideContext)", add_bos: false)
        let startOffset = prompt_tokens.count - 1

        guard let logits = self.get_logits(tokens: prompt_tokens, logits_start_index: startOffset) else {
            debug("logits unavailable")
            return []
        }

        let n_vocab = llama_vocab_n_tokens(vocab)
        var exp_sum: Float = 0
        let startIndex = (prompt_tokens.count - 1 - startOffset) * Int(n_vocab)
        let endIndex = (prompt_tokens.count - startOffset) * Int(n_vocab)

        // Min-Heapを使用してn-bestを計算
        var minHeap: FixedSizeHeap<NextCharacterCandidate> = .init(size: count)
        let token_to_penalty_weight: [llama_token: Float] = prompt_tokens.indexed().reduce(into: [:]) { dict, item in
            let (index, token) = item
            // 現在位置から遠いほど減衰させる
            dict[token, default: 0] += 2 / Float(prompt_tokens.count - index)
        }

        for index in startIndex..<endIndex {
            let token = llama_token(index - startIndex)
            let repeat_penalty = Float(1.0 + token_to_penalty_weight[token, default: 0])
            let v = expf(logits[index] / repeat_penalty)
            exp_sum += v

            let tokenPieceData = Data((token_to_piece(token: token)).map(UInt8.init))
            let character: Character
            if let validCharacter = String(data: tokenPieceData, encoding: .utf8), let c = validCharacter.first {
                character = c
            } else {
                continue
            }
            minHeap.insertIfPossible(NextCharacterCandidate(character: character, value: v))
        }

        // Heapからソートして結果を取り出す
        return minHeap.unordered.sorted { $0.value > $1.value }.map { ($0.character, $0.value / exp_sum) }
    }

    func evaluate_candidate(
        input: String,
        candidate: Candidate,
        requestRichCandidates: Bool,
        prefixConstraint: Kana2Kanji.PrefixConstraint,
        personalizationMode: (mode: ConvertRequestOptions.ZenzaiMode.PersonalizationMode, base: EfficientNGram, personal: EfficientNGram)?,
        versionDependentConfig: ConvertRequestOptions.ZenzaiVersionDependentMode
    ) -> CandidateEvaluationResult {
        debug("Evaluate", candidate)
        // For zenz-v1 model, \u{EE00} is a token used for 'start query', and \u{EE01} is a token used for 'start answer'
        // We assume \u{EE01}\(candidate) is always splitted into \u{EE01}_\(candidate) by zenz-v1 tokenizer
        var userDictionaryPrompt: String = ""
        for item in candidate.data where item.metadata.contains(.isFromUserDictionary) {
            userDictionaryPrompt += "\(item.word)(\(item.ruby.toHiragana()))"
        }
        var conditions: [String] = []
        // ユーザ辞書の内容がある場合はこれを条件に追加
        if !userDictionaryPrompt.isEmpty {
            conditions.append("辞書:\(userDictionaryPrompt)")
        }
        // プロフィールがある場合はこれを条件に追加
        switch versionDependentConfig {
        case .v1: break
        case .v2(let mode):
            if let profile = mode.profile, !profile.isEmpty {
                let pf = profile.suffix(25)
                conditions.append("プロフィール:\(pf)")
            }
        case .v3(let mode):
            if let profile = mode.profile, !profile.isEmpty {
                let pf = profile.suffix(25)
                conditions.append("\u{EE03}\(pf)")
            }
            if let topic = mode.topic, !topic.isEmpty {
                let tp = topic.suffix(25)
                conditions.append("\u{EE04}\(tp)")
            }
            if let style = mode.style, !style.isEmpty {
                let st = style.suffix(25)
                conditions.append("\u{EE05}\(st)")
            }
            if let preference = mode.preference, !preference.isEmpty {
                let pr = preference.suffix(25)
                conditions.append("\u{EE06}\(pr)")
            }
        }
        // 左文脈を取得
        let leftSideContext: String = switch versionDependentConfig {
        case .v1: ""
        case .v2(let mode):
            if let leftSideContext = mode.leftSideContext {
                String(leftSideContext.suffix(mode.maxLeftSideContextLength ?? 40))
            } else {
                ""
            }
        case .v3(let mode):
            if let leftSideContext = mode.leftSideContext {
                String(leftSideContext.suffix(mode.maxLeftSideContextLength ?? 40))
            } else {
                ""
            }
        }
        let inputTag = "\u{EE00}"
        let outputTag = "\u{EE01}"
        let contextTag = "\u{EE02}"
        // プロンプトを作成
        var prompt: String = switch versionDependentConfig {
        case .v1:
            inputTag + input + outputTag
        case .v2:
            if !conditions.isEmpty {
                // 条件がemptyでない場合は「・」でつなぎ、「発言:」を末尾に追加
                inputTag + input + contextTag + conditions.joined(separator: "・") + "・発言:\(leftSideContext)" + outputTag
            } else if !leftSideContext.isEmpty {
                // 条件がemptyの場合、単にleftSideContextを追加
                inputTag + input + contextTag + leftSideContext + outputTag
            } else {
                // そのまま
                inputTag + input + outputTag
            }
        case .v3:
            if !leftSideContext.isEmpty {
                // leftSideContextがEmptyでなければ下記の通り処理
                // contextがinputに前置されるように変更された(KV-cachingの効率化のため)
                conditions.joined(separator: "") + contextTag + leftSideContext + inputTag + input + outputTag
            } else {
                // そのまま
                conditions.joined(separator: "") + inputTag + input + outputTag
            }
        }
        // プロンプトの前処理を適用
        prompt = self.preprocessText(text: prompt)
        // Therefore, tokens = prompt_tokens + candidate_tokens is an appropriate operation.
        let prompt_tokens = self.tokenize(text: prompt, add_bos: true, add_eos: false)
        defer {
            self.prevPrompt = prompt_tokens
        }

        let candidate_tokens = self.tokenize(text: self.preprocessText(text: candidate.text), add_bos: false, add_eos: false)
        // prefixConstraintをすでに満たしているトークンを調査する
        let addressed_tokens: [llama_token]
        if self.prevPrompt == prompt_tokens, !requestRichCandidates {
            var string = ""
            for character in candidate.text {
                let newString = string + String(character)
                if prefixConstraint.constraint.hasPrefix(newString.utf8) {
                    string = newString
                } else {
                    break
                }
            }
            // addressedTokensについてはそのまま扱えばよい
            addressed_tokens = self.tokenize(text: self.preprocessText(text: string), add_bos: false, add_eos: false)
        } else {
            // rich candidatesのため、logit全体を得る必要がある
            addressed_tokens = []
        }

        let tokens = prompt_tokens + candidate_tokens

        // すでにprefixConstraintを満たしている部分については、計算をしない
        let startOffset = prompt_tokens.count - 1 + addressed_tokens.count
        guard let logits = self.get_logits(tokens: tokens, logits_start_index: startOffset) else {
            debug("logits unavailable")
            return .error
        }
        let n_vocab = llama_vocab_n_tokens(vocab)
        let is_learned_token: [(isLearned: Bool, priority: Float)] = Array(repeating: (false, 0), count: prompt_tokens.count) + candidate.data.flatMap {
            // priorityは文字数にする→文字数が長いほど優先される
            Array(repeating: ($0.metadata.contains(.isLearned), logf(getLearningPriority(data: $0))), count: self.tokenize(text: $0.word, add_bos: false).count)
        }

        var score: Float = 0

        struct AlternativeHighProbToken: Comparable {
            static func < (lhs: AlternativeHighProbToken, rhs: AlternativeHighProbToken) -> Bool {
                lhs.probabilityRatioToMaxProb < rhs.probabilityRatioToMaxProb
            }

            var token: llama_token
            var constraint: [UInt8]
            // 最大probabilityに対しての割合
            var probabilityRatioToMaxProb: Float
        }

        var altTokens = FixedSizeHeap<AlternativeHighProbToken>(size: requestRichCandidates ? 5 : 0)
        for (i, token_id) in tokens.indexed().dropFirst(startOffset + 1) {
            // それぞれのトークンが、一つ前の予測において最も確率の高いトークンであるかをチェックする
            // softmaxはmaxなので、単にlogitsの中で最も大きいものを選べば良い
            // 一方実用的にはlog_probも得ておきたい。このため、ここでは明示的にsoftmaxも計算している
            struct TokenAndLogprob: Comparable {
                static func < (lhs: TokenAndLogprob, rhs: TokenAndLogprob) -> Bool {
                    lhs.logprob < rhs.logprob
                }
                var token: llama_token
                var logprob: Float
            }
            var sumexp: Float = 0
            let startIndex = (i - 1 - startOffset) * Int(n_vocab)
            let endIndex = (i - startOffset) * Int(n_vocab)
            var tokenHeap = FixedSizeHeap<TokenAndLogprob>(size: requestRichCandidates ? 3 : 1)
            for index in startIndex ..< endIndex {
                sumexp += expf(logits[index])
            }
            let logsumexp = logf(sumexp)

            if let (mode, baseLM, personalLM) = personalizationMode, mode.alpha > 0 {
                let prefix = tokens[..<i].dropFirst(prompt_tokens.count).map(Int.init)
                let baseProb: [Float]
                let personalProb: [Float]
                // SwiftNgramのLMは無条件の場合エラーになるため(Unigram確率はサポートしていない)
                if !prefix.isEmpty {
                    baseProb = baseLM.bulkPredict(prefix).map { logf(Float($0) + 1e-7) }
                    personalProb = personalLM.bulkPredict(prefix).map { logf(Float($0) + 1e-7) }
                } else {
                    baseProb = Array(repeating: 0, count: Int(n_vocab))
                    personalProb = baseProb
                }
                // p = probabilityBuffer / exp_sum
                // p' = p / p_b * p_p
                for (i, (lpb, lpp)) in zip(0 ..< Int(n_vocab), zip(baseProb, personalProb)) {
                    let logp = logits[startIndex + i] - logsumexp
                    let logp_ = logp + mode.alpha * (lpp - lpb) // personalized probability
                    tokenHeap.insertIfPossible(TokenAndLogprob(token: llama_token(i), logprob: logp_))
                }
            } else {
                // p = probabilityBuffer / exp_sum
                for i in startIndex ..< endIndex {
                    let logp = logits[i] - logsumexp
                    tokenHeap.insertIfPossible(TokenAndLogprob(token: llama_token(i - startIndex), logprob: logp))
                }
            }

            guard let maxItem = tokenHeap.max else {
                debug("Max Item could not be found for unknown reason")
                return .error
            }
            // ここで最も良い候補であったかをチェックする
            if maxItem.token != token_id {
                if maxItem.token == llama_vocab_eos(vocab) {
                    let cchars: [CChar] = tokens[..<i].reduce(into: []) {
                        $0.append(contentsOf: token_to_piece(token: $1))
                    }
                    let data = Data(cchars.map { UInt8(bitPattern: $0) })
                    let string: String = String(data: data, encoding: .utf8) ?? ""
                    // 要求するべき制約を記述する
                    let wholeResult = String(string.dropFirst(prompt.count))
                    return .wholeResult(wholeResult)
                } else {
                    let actual_logp: Float = logits[startIndex + Int(token_id)] - logsumexp
                    // 学習されたトークンであり、なおかつactual_expのある程度大きければ、学習されたトークンを優先する
                    let preferLearnedToken = is_learned_token[i].isLearned && actual_logp + is_learned_token[i].priority > maxItem.logprob
                    if !preferLearnedToken {
                        // adding "\0"
                        let cchars = tokens[..<i].reduce(into: []) {
                            $0.append(contentsOf: token_to_piece(token: $1))
                        } + token_to_piece(token: maxItem.token)
                        return .fixRequired(prefixConstraint: cchars.dropFirst(prompt.utf8.count).map(UInt8.init))
                    }
                }
            } else if !tokenHeap.isEmpty {
                tokenHeap.removeMax()
                let prefix = tokens[..<i].reduce(into: []) {
                    $0.append(contentsOf: token_to_piece(token: $1))
                }.dropFirst(prompt.utf8.count)

                for item in tokenHeap.unordered {
                    altTokens.insertIfPossible(
                        AlternativeHighProbToken(
                            token: item.token,
                            constraint: prefix.map(UInt8.init) + token_to_piece(token: item.token).map(UInt8.init),
                            probabilityRatioToMaxProb: expf(item.logprob - maxItem.logprob)
                        )
                    )
                }
            }
            score += maxItem.logprob
        }
        return .pass(score: score, alternativeConstraints: altTokens.unordered.sorted(by: >).map {.init(probabilityRatio: $0.probabilityRatioToMaxProb, prefixConstraint: $0.constraint)})
    }

    private func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], logits: Bool) {
        batch.token   [Int(batch.n_tokens)] = id
        batch.pos     [Int(batch.n_tokens)] = pos
        batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
        for i in 0..<seq_ids.count {
            batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
        }
        batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0
        batch.n_tokens += 1
    }

    private func preprocessText(text: String) -> String {
        // replace space into ideographic space (\u3000) for zenz tokenizer
        // replace newline into null for zenz tokenizer
        text.replacingOccurrences(of: " ", with: "\u{3000}").replacingOccurrences(of: "\n", with: "")
    }
    private func tokenize(text: String, add_bos: Bool, add_eos: Bool = false) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0)
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, false)
        var swiftTokens: [llama_token] = if tokenCount < 0 {
            [llama_vocab_bos(vocab)]
        } else {
            (0..<tokenCount).map {tokens[Int($0)]}
        }
        tokens.deallocate()
        if add_eos {
            swiftTokens.append(llama_vocab_eos(vocab))
        }
        return swiftTokens
    }

    /// - note: The result does not contain null-terminator
    private func token_to_piece(token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(vocab, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(vocab, token, newResult, Int32(-nTokens), 0, false)
            let bufferPointer: UnsafeBufferPointer<Int8> = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer: UnsafeBufferPointer<Int8> = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
}

#if Zenzai
extension ZenzContext {
    private func nativeError() -> NSError {
        NSError(domain: "HazkeyNativeTTT", code: -1,
                userInfo: [NSLocalizedDescriptionKey: String(cString: llama_ttt_last_error())])
    }
    func nativeApply(_ document: String) throws {
        guard llama_ttt_apply(context, document, 1e-4) == 0 else { throw nativeError() }
        prevInput = []
        prevPrompt = []
    }
    func nativeWord(prefix: String, output: String, maxSteps: Int) throws -> (Bool, Int) {
        var steps: Int32 = 0
        let result = llama_ttt_learn_word(context, prefix, output, Int32(maxSteps), &steps)
        prevInput = []
        prevPrompt = []
        guard result >= 0 else { throw nativeError() }
        return (result == 1, Int(steps))
    }
    func nativeSave(path: String, cursor: UInt64) throws {
        guard llama_ttt_save(context, path, cursor) == 0 else { throw nativeError() }
    }
    func nativeLoad(path: String) throws -> UInt64 {
        var cursor: UInt64 = 0
        guard llama_ttt_load(context, path, &cursor) == 0 else { throw nativeError() }
        prevInput = []
        prevPrompt = []
        return cursor
    }
}
#endif

#if Zenzai
public final class NativeZenzaiTTTVersion: @unchecked Sendable {
    fileprivate let handle: OpaquePointer
    fileprivate init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { llama_ttt_free_version(handle) }
}
extension ZenzContext {
    func nativeExport() throws -> NativeZenzaiTTTVersion {
        guard let version = llama_ttt_export_version(context) else { throw nativeError() }
        return NativeZenzaiTTTVersion(version)
    }
    func nativeImport(_ version: NativeZenzaiTTTVersion) throws {
        guard llama_ttt_import_version(context, version.handle) == 0 else { throw nativeError() }
        prevInput = []
        prevPrompt = []
    }
}
#endif
