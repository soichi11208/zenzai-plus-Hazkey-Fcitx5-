import EfficientNGram
package import Foundation
import SwiftUtils

package final class Zenz {
    package var resourceURL: URL
    private var zenzContext: ZenzContext?

    init(resourceURL: URL, deviceConfig: ZenzaiDeviceConfig = ZenzaiDeviceConfig(), contextLength: Int32 = 512) throws {
        self.resourceURL = resourceURL
        do {
            #if canImport(Darwin)
            if #available(iOS 16, macOS 13, *) {
                self.zenzContext = try ZenzContext.createContext(path: resourceURL.path(percentEncoded: false), deviceConfig: deviceConfig, contextLength: contextLength)
            } else {
                // this is not percent-encoded
                self.zenzContext = try ZenzContext.createContext(path: resourceURL.path, deviceConfig: deviceConfig, contextLength: contextLength)
            }
            #else
            // this is not percent-encoded
            self.zenzContext = try ZenzContext.createContext(path: resourceURL.path, deviceConfig: deviceConfig, contextLength: contextLength)
            #endif
            debug("Loaded model \(resourceURL.lastPathComponent) with device config: \(deviceConfig)")
        } catch {
            throw error
        }
    }

    package func endSession() {
        try? self.zenzContext?.reset_context()
    }

    /// Update device configuration dynamically
    public func updateDeviceConfig(_ newConfig: ZenzaiDeviceConfig) throws {
        try self.zenzContext?.updateDeviceConfig(newConfig)
    }

    /// Get current device configuration
    public func getDeviceConfig() -> ZenzaiDeviceConfig? {
        return self.zenzContext?.getDeviceConfig()
    }

    func candidateEvaluate(
        convertTarget: String,
        candidates: [Candidate],
        requestRichCandidates: Bool,
        prefixConstraint: Kana2Kanji.PrefixConstraint,
        personalizationMode: (mode: ConvertRequestOptions.ZenzaiMode.PersonalizationMode, base: EfficientNGram, personal: EfficientNGram)?,
        versionDependentConfig: ConvertRequestOptions.ZenzaiVersionDependentMode
    ) -> ZenzContext.CandidateEvaluationResult {
        guard let zenzContext else {
            return .error
        }
        for candidate in candidates {
            return zenzContext.evaluate_candidate(
                input: convertTarget.toKatakana(),
                candidate: candidate,
                requestRichCandidates: requestRichCandidates,
                prefixConstraint: prefixConstraint,
                personalizationMode: personalizationMode,
                versionDependentConfig: versionDependentConfig
            )
        }
        return .error
    }

    func predictNextCharacter(leftSideContext: String, count: Int) -> [(character: Character, value: Float)] {
        guard let zenzContext else {
            return []
        }
        return zenzContext.predict_next_character(leftSideContext: leftSideContext, count: count)
    }

    package func pureGreedyDecoding(pureInput: String, maxCount: Int = .max) -> String {
        self.zenzContext?.pure_greedy_decoding(leftSideContext: pureInput, maxCount: maxCount) ?? ""
    }
}

#if Zenzai
extension Zenz {
    package func nativeApply(_ document: String) throws { try nativeContext().nativeApply(document) }
    package func nativeWord(prefix: String, output: String, maxSteps: Int) throws -> (Bool, Int) {
        try nativeContext().nativeWord(prefix: prefix, output: output, maxSteps: maxSteps)
    }
    package func nativeSave(path: String, cursor: UInt64) throws {
        try nativeContext().nativeSave(path: path, cursor: cursor)
    }
    package func nativeLoad(path: String) throws -> UInt64 { try nativeContext().nativeLoad(path: path) }
    private func nativeContext() throws -> ZenzContext {
        guard let context = zenzContext else {
            throw NSError(domain: "HazkeyNativeTTT", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Zenzai context is unavailable"])
        }
        return context
    }
}
#endif

#if Zenzai
extension Zenz {
    package func nativeExport() throws -> NativeZenzaiTTTVersion { try nativeContext().nativeExport() }
    package func nativeImport(_ version: NativeZenzaiTTTVersion) throws { try nativeContext().nativeImport(version) }
}
#endif
