import Foundation
import KanaKanjiConverterModule
import SwiftUtils

/// A regular dictionary merge: no user-dictionary flags or forced ranking.
final class NicoDictionary {
    private(set) var directory: URL?

    init(dictionaryURL: URL) {
        var source = HazkeyServerConfig.getDataDirectory()
            .appendingPathComponent("dictionaries/dic-nico-intersection-pixiv-google.txt")
        let manager = FileManager.default
        if !manager.fileExists(atPath: source.path) {
            source = URL(fileURLWithPath: systemResourcePath)
                .appendingPathComponent("dictionaries/dic-nico-intersection-pixiv-google.txt")
        }
        guard manager.fileExists(atPath: source.path) else { return }
        do {
            let raw = try Data(contentsOf: source)
            let originalLOUDS = dictionaryURL.appendingPathComponent("louds")
            let originalFiles = try manager.contentsOfDirectory(at: originalLOUDS,
                includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey]).sorted { $0.path < $1.path }
            var fingerprint = raw
            fingerprint.append(try Data(contentsOf: originalLOUDS.appendingPathComponent("charID.chid")))
            for file in originalFiles {
                let info = try file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
                fingerprint.append(Data("\(file.lastPathComponent):\(info.fileSize ?? 0):\(info.contentModificationDate?.timeIntervalSince1970 ?? 0)".utf8))
            }
            var hash: UInt64 = 14695981039346656037
            for byte in fingerprint { hash = (hash ^ UInt64(byte)) &* 1099511628211 }
            let cache = HazkeyServerConfig.getCacheDirectory()
                .appendingPathComponent("nico-regular-merged-v1-" + String(hash, radix: 16))
            if !manager.fileExists(atPath: cache.appendingPathComponent("complete").path) {
                var entries: [DicdataElement] = []
                var seen = Set<String>()
                for file in originalFiles where file.pathExtension == "loudstxt3" {
                    let data = try Data(contentsOf: file)
                    try Self.decode(data) { entry in
                        // Preserve every original POS variant and score.
                        entries.append(entry)
                        seen.insert(entry.ruby + "\u{0}" + entry.word)
                    }
                }
                let originalCount = entries.count
                for line in String(decoding: raw, as: UTF8.self).split(whereSeparator: \.isNewline)
                    where !line.hasPrefix("#") {
                    let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
                    guard fields.count >= 2, !fields[0].isEmpty, !fields[1].isEmpty else { continue }
                    let ruby = String(fields[0]).toKatakana()
                    let word = String(fields[1])
                    guard seen.insert(ruby + "\u{0}" + word).inserted else { continue }
                    // Median of the original dictionary's proper-noun scores.
                    entries.append(DicdataElement(word: word, ruby: ruby,
                        cid: CIDData.固有名詞.cid, mid: MIDData.一般.mid, value: -13.8586))
                }
                let staging = cache.appendingPathExtension(UUID().uuidString)
                let louds = staging.appendingPathComponent("louds")
                try manager.createDirectory(at: louds, withIntermediateDirectories: true)
                do {
                    for file in try manager.contentsOfDirectory(at: dictionaryURL, includingPropertiesForKeys: nil)
                        where file.lastPathComponent != "louds" {
                        try manager.createSymbolicLink(at: staging.appendingPathComponent(file.lastPathComponent),
                                                       withDestinationURL: file)
                    }
                    // Preserve auxiliary files; rebuild only the trie and its entry shards.
                    for file in originalFiles where !["louds", "loudschars2", "loudstxt3"].contains(file.pathExtension) {
                        try manager.copyItem(at: file, to: louds.appendingPathComponent(file.lastPathComponent))
                    }
                    try DictionaryBuilder.exportDictionary(entries: entries, to: louds,
                        baseName: "", shardByFirstCharacter: true,
                        charIDFileURL: originalLOUDS.appendingPathComponent("charID.chid"))
                    let manifest: [String: Any] = ["original_entries": originalCount,
                        "added_entries": entries.count - originalCount, "total_entries": entries.count,
                        "priority": "none", "new_entry_score": -13.8586]
                    try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted])
                        .write(to: staging.appendingPathComponent("complete"))
                    if manager.fileExists(atPath: cache.path) { try manager.removeItem(at: cache) }
                    try manager.moveItem(at: staging, to: cache)
                } catch {
                    try? manager.removeItem(at: staging)
                    throw error
                }
            }
            directory = cache
            NSLog("Loaded nico-pixiv as regular dictionary: %@", cache.path)
        } catch {
            NSLog("Failed to merge nico-pixiv dictionary: \(error.localizedDescription)")
        }
    }

    private static func decode(_ data: Data, append: (DicdataElement) -> Void) throws {
        enum InvalidDictionary: Error { case invalidData }
        try data.withUnsafeBytes { bytes in
            func uint16(_ offset: Int) throws -> Int {
                guard offset >= 0, offset + 2 <= bytes.count else { throw InvalidDictionary.invalidData }
                return Int(UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
            }
            let slots = try uint16(0)
            guard 2 + slots * 4 <= bytes.count else { throw InvalidDictionary.invalidData }
            let offsets = (0..<slots).map {
                Int(UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: 2 + $0 * 4, as: UInt32.self)))
            }
            for (index, offset) in offsets.enumerated() {
                let end = index + 1 < slots ? offsets[index + 1] : bytes.count
                let count = try uint16(offset)
                if count == 0 { continue }
                let start = offset + 2 + count * 10
                guard start <= end, end <= bytes.count else { throw InvalidDictionary.invalidData }
                let text = String(decoding: data[start..<end], as: UTF8.self)
                    .split(separator: "\t", omittingEmptySubsequences: false)
                guard text.count == count + 1 else { throw InvalidDictionary.invalidData }
                let ruby = String(text[0])
                for row in 0..<count {
                    let position = offset + 2 + row * 10
                    let scoreBits = UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: position + 6, as: UInt32.self))
                    append(DicdataElement(word: text[row + 1].isEmpty ? ruby : String(text[row + 1]),
                        ruby: ruby, lcid: try uint16(position), rcid: try uint16(position + 2),
                        mid: try uint16(position + 4), value: Float(bitPattern: scoreBits)))
                }
            }
        }
    }
}
