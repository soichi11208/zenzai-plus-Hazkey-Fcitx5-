import Foundation
import Glibc
import KanaKanjiConverterModule

/// Feeds confirmed input into the same llama.cpp model used for conversion.
final class ZenzaiTTTAdapter: @unchecked Sendable {
    static let shared = ZenzaiTTTAdapter()
    private let lock = NSRecursiveLock()
    private var dataDirectory: URL?
    private var timer: DispatchSourceTimer?
    private var workerLock: Int32 = -1
    private var onTick: (@Sendable () -> Void)?
    private var loadedSnapshot: String?
    private var processed: UInt64 = 0
    // Only the background queue touches the trainer and loadedSnapshot.
    private var trainer: NativeZenzaiTTTTrainer?
    private var publishedVersion: NativeZenzaiTTTVersion?
    private var publicationKey: String?
    private var publicationNumber: UInt64 = 0
    private var appliedNumber: UInt64 = 0
    private var resetPending = false
    private var retryAfter = Date.distantPast
    private init() {}

    func start(dataDirectory: URL, update: (@Sendable () -> Void)? = nil) {
        lock.lock()
        defer { lock.unlock() }
        self.dataDirectory = dataDirectory.appendingPathComponent("zenzai-ttt", isDirectory: true)
        if let update { self.onTick = update }
        guard timer == nil, let root = self.dataDirectory else { return }
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
        } catch {
            NSLog("Cannot create native TTT directory: %@", error.localizedDescription)
            return
        }
        let fd = open(root.appendingPathComponent("worker.lock").path, O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
        guard fd >= 0 else { NSLog("Cannot open native TTT lock"); return }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            close(fd)
            NSLog("Native TTT not started: another adaptation worker holds the lock")
            return
        }
        workerLock = fd
        NSLog("Native llama.cpp TTT enabled; learning and snapshots run outside the input thread")
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "hazkey.native-ttt", qos: .utility))
        timer.schedule(deadline: .now() + 1, repeating: 1)
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            self.lock.lock()
            let callback = self.onTick
            self.lock.unlock()
            callback?()
        }
        self.timer = timer
        timer.resume()
    }

    func activeModelURL(fallback: URL) -> URL {
        lock.lock()
        defer { lock.unlock() }
        guard let root = dataDirectory,
              let text = try? String(contentsOf: root.appendingPathComponent("active-model.txt"), encoding: .utf8)
        else { return fallback }
        if FileManager.default.fileExists(atPath: root.appendingPathComponent("reset").path) {
            return root.appendingPathComponent("base.gguf")
        }
        let path = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return !path.isEmpty && FileManager.default.fileExists(atPath: path) ? URL(fileURLWithPath: path) : fallback
    }

    func submit(reading: String, context: String, output: String) {
        lock.lock()
        defer { lock.unlock() }
        guard !reading.isEmpty, !output.isEmpty, let root = dataDirectory else { return }
        do {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            let url = root.appendingPathComponent("examples.jsonl")
            if !FileManager.default.fileExists(atPath: url.path) {
                _ = FileManager.default.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600])
            }
            let data = try JSONSerialization.data(withJSONObject: ["reading": reading, "context": context, "output": output])
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: data + Data([10]))
            try handle.synchronize()
        } catch { NSLog("Failed to queue native TTT: %@", error.localizedDescription) }
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        guard let root = dataDirectory else { return }
        resetPending = true
        do { try Data().write(to: root.appendingPathComponent("reset"), options: .atomic) }
        catch { NSLog("Failed to request native TTT reset: %@", error.localizedDescription) }
    }

    private func json(_ url: URL) throws -> [String: Any] {
        guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw NSError(domain: "HazkeyNativeTTT", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid JSON: \(url.lastPathComponent)"])
        }
        return value
    }

    private func write(_ value: Any, to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    private func snapshot(root: URL, model: URL) throws -> URL {
        let attributes = try FileManager.default.attributesOfItem(atPath: model.path)
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let identity = "\(model.resolvingSymlinksInPath().path):\(attributes[.size] ?? 0):\(modified)"
        var hash: UInt64 = 14695981039346656037
        for byte in identity.utf8 { hash = (hash ^ UInt64(byte)) &* 1099511628211 }
        return root.appendingPathComponent("native-ttt-\(String(hash, radix: 16)).bin")
    }

    private func register(_ row: [String: Any], root: URL) throws {
        guard let reading = row["reading"] as? String, let output = row["output"] as? String,
              !reading.isEmpty, !output.isEmpty, reading.count <= 128, output.count <= 128 else {
            throw NSError(domain: "HazkeyNativeTTT", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "読みと単語を1〜128文字で指定してください"])
        }
        let url = root.appendingPathComponent("registered-words.json")
        var words = (try? JSONSerialization.jsonObject(with: Data(contentsOf: url))) as? [[String: String]] ?? []
        let normalized = reading.applyingTransform(.hiraganaToKatakana, reverse: false) ?? reading
        let entry = ["reading": normalized, "output": output]
        if !words.contains(entry) { words.append(entry); try write(words, to: url) }
    }


    private func offer(_ trainer: NativeZenzaiTTTTrainer, key: String) throws {
        let version = try trainer.exportVersion()
        lock.lock()
        let previous = publishedVersion
        publishedVersion = version
        publicationKey = key
        publicationNumber += 1
        lock.unlock()
        withExtendedLifetime(previous) {}
    }

    /// Called by the input thread. It never waits for training or snapshot I/O.
    func publish(converter: KanaKanjiConverter, options: ConvertRequestOptions, modelURL: URL, deviceKey: String) {
        lock.lock()
        let root = dataDirectory
        let reset = resetPending
        resetPending = false
        lock.unlock()
        if reset { converter.clearNativeTTTModel() }
        guard let root, let checkpoint = try? snapshot(root: root, model: modelURL) else { return }
        let key = checkpoint.path + "@" + deviceKey
        lock.lock()
        let version = publicationKey == key && publicationNumber != appliedNumber ? publishedVersion : nil
        let number = publicationNumber
        lock.unlock()
        guard let version else { return }
        do {
            try converter.publishNativeTTT(version, options: options)
            lock.lock()
            appliedNumber = number
            lock.unlock()
        } catch { NSLog("Failed to publish native TTT: %@", error.localizedDescription) }
    }

    /// Runs only on the background queue, with no input-thread model lock held.
    func process(options: ConvertRequestOptions, modelURL: URL, deviceKey: String) {
        lock.lock()
        let directory = dataDirectory
        lock.unlock()
        guard let root = directory, workerLock >= 0, Date() >= retryAfter else { return }
        let manager = FileManager.default
        do {
            let resetFlag = root.appendingPathComponent("reset")
            if manager.fileExists(atPath: resetFlag.path) {
                trainer = nil
                loadedSnapshot = nil
                lock.lock()
                publishedVersion = nil
                publicationKey = nil
                resetPending = true
                lock.unlock()
                for file in try manager.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
                    where file.lastPathComponent.hasPrefix("native-ttt-") {
                    try manager.removeItem(at: file)
                }
                try? manager.removeItem(at: root.appendingPathComponent("active-model.txt"))
                try Data().write(to: root.appendingPathComponent("examples.jsonl"), options: .atomic)
                try write(["processed": 0], to: root.appendingPathComponent("progress.json"))
                try manager.removeItem(at: resetFlag)
                processed = 0
                return
            }
            let requests = root.appendingPathComponent("word-requests")
            let pending = ((try? manager.contentsOfDirectory(at: requests, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension == "json" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
            let queue = root.appendingPathComponent("examples.jsonl")
            let data = (try? Data(contentsOf: queue)) ?? Data()
            let rows = data.split(separator: 10).compactMap {
                (try? JSONSerialization.jsonObject(with: Data($0))) as? [String: String]
            }.filter { !($0["reading"] ?? "").isEmpty && !($0["output"] ?? "").isEmpty }
            let savedProgress = (try? json(root.appendingPathComponent("progress.json"))["processed"] as? NSNumber)?.uint64Value ?? 0
            let checkpoint = try snapshot(root: root, model: modelURL)
            let sessionKey = checkpoint.path + "@" + deviceKey
            if loadedSnapshot != sessionKey {
                trainer = nil
                lock.lock()
                publishedVersion = nil
                publicationKey = nil
                lock.unlock()
                guard manager.fileExists(atPath: checkpoint.path) || !pending.isEmpty || UInt64(rows.count) > savedProgress else { return }
                let next = try NativeZenzaiTTTTrainer(options: options)
                if manager.fileExists(atPath: checkpoint.path) {
                    processed = try next.load(path: checkpoint.path)
                    try offer(next, key: sessionKey)
                } else { processed = savedProgress }
                trainer = next
                loadedSnapshot = sessionKey
            }
            guard let trainer, !pending.isEmpty || UInt64(rows.count) > processed else { return }
            if let request = pending.first {
                let status = request.deletingPathExtension().appendingPathExtension("status")
                do {
                    let row = try json(request)
                    try register(row, root: root)
                    try write(["state": "training", "steps": 0, "dictionary_registered": true], to: status)
                    let reading = row["reading"] as! String
                    let output = row["output"] as! String
                    let kana = reading.applyingTransform(.hiraganaToKatakana, reverse: false) ?? reading
                    let limit = min(128, max(1, (row["max_steps"] as? Int) ?? 32))
                    let started = Date()
                    let (matched, steps) = try trainer.learnWord(prefix: "\u{EE00}" + kana + "\u{EE01}",
                        output: output, maxSteps: limit)
                    let elapsed = Date().timeIntervalSince(started)
                    if matched && steps > 0 {
                        try trainer.save(path: checkpoint.path, cursor: processed)
                        try offer(trainer, key: sessionKey)
                    }
                    try write(["state": "done", "steps": steps, "matched": matched,
                        "dictionary_registered": true, "method": "llama.cpp-native-final-mlp-gradient",
                        "training_seconds": elapsed, "model_updated": matched && steps > 0], to: status)
                    NSLog("Native word learning: matched=%@, steps=%d, train=%.3fs", String(matched), steps, elapsed)
                } catch {
                    try? write(["state": "error", "message": error.localizedDescription], to: status)
                    try? manager.moveItem(at: request, to: request.deletingPathExtension().appendingPathExtension("done"))
                    throw error
                }
                try manager.moveItem(at: request, to: request.deletingPathExtension().appendingPathExtension("done"))
                return
            }
            let modified = try queue.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? Date()
            guard Date().timeIntervalSince(modified) >= 2 else { return }
            var seen = Set<String>()
            let updates = rows.dropFirst(Int(processed)).reversed().filter {
                let ruby = ($0["reading"] ?? "").applyingTransform(.hiraganaToKatakana, reverse: false) ?? $0["reading"]!
                return seen.insert(ruby).inserted
            }.reversed()
            let started = Date()
            for row in updates {
                let ruby = row["reading"]!.applyingTransform(.hiraganaToKatakana, reverse: false) ?? row["reading"]!
                var document = "\u{EE00}" + ruby
                if let context = row["context"], !context.isEmpty { document += "\u{EE02}" + String(String.UnicodeScalarView(context.unicodeScalars.suffix(40))) }
                document += "\u{EE01}" + row["output"]!
                try trainer.apply(document: document)
            }
            let next = UInt64(rows.count)
            try trainer.save(path: checkpoint.path, cursor: next)
            try offer(trainer, key: sessionKey)
            processed = next
            try write(["processed": processed, "method": "llama.cpp-native"], to: root.appendingPathComponent("progress.json"))
            NSLog("Native TTT updated through %llu conversions in %.3fs", processed, Date().timeIntervalSince(started))
        } catch {
            trainer = nil
            loadedSnapshot = nil
            retryAfter = Date().addingTimeInterval(10)
            NSLog("Native TTT failed: %@", error.localizedDescription)
        }
    }
}
