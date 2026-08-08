import Foundation

/// Minimal SentencePiece unigram tokenizer for PocketTTS.
///
/// Parses a `.model` protobuf to extract the vocabulary, then uses
/// Viterbi decoding to segment text into subword tokens.
public struct SentencePieceTokenizer: Sendable {

    /// Vocabulary pieces with their log-probability scores.
    private let pieces: [SentencePieceProto.Piece]
    /// Lookup from piece string to token ID.
    private let pieceToId: [String: Int]
    /// Lookup from a raw UTF-8 byte to its SentencePiece byte-fallback token ID.
    private let bytePieceIds: [UInt8: Int]
    /// Whether unknown scalars must expand to UTF-8 byte tokens.
    private let byteFallbackEnabled: Bool
    /// Token ID used for an unknown scalar before optional byte expansion.
    private let unknownPieceId: Int?
    /// Score assigned to unknown scalar edges during Viterbi decoding.
    private let unknownScore: Float
    /// Maximum piece length in Unicode scalars for early termination.
    private let maxPieceLength: Int

    /// The space replacement character used by SentencePiece.
    private static let spaceMarker: Character = "\u{2581}"
    private static let unknownPenalty: Float = 10

    public init(modelData: Data) throws {
        let model = try SentencePieceProto.parseModel(modelData)
        self.pieces = model.pieces

        var lookup: [String: Int] = [:]
        lookup.reserveCapacity(model.pieces.count)
        var byteLookup: [UInt8: Int] = [:]
        var maxLen = 0
        var unknownId: Int?
        var unknownCount = 0
        var minimumNormalScore: Float?
        for (index, entry) in model.pieces.enumerated() {
            switch entry.type {
            case .normal:
                lookup[entry.piece] = index
                maxLen = max(maxLen, entry.piece.unicodeScalars.count)
                minimumNormalScore = min(minimumNormalScore ?? entry.score, entry.score)
            case .unknown:
                unknownId = index
                unknownCount += 1
            case .userDefined:
                lookup[entry.piece] = index
                maxLen = max(maxLen, entry.piece.unicodeScalars.count)
            case .byte:
                if let byte = Self.byteValue(for: entry.piece) {
                    byteLookup[byte] = index
                }
            case .control, .unused:
                break
            }
        }

        if model.byteFallbackEnabled {
            guard unknownCount == 1, byteLookup.count == 256 else {
                throw SentencePieceProto.ParseError.invalidData
            }
        }

        self.pieceToId = lookup
        self.bytePieceIds = byteLookup
        self.byteFallbackEnabled = model.byteFallbackEnabled
        self.unknownPieceId = unknownId
        self.unknownScore = (minimumNormalScore ?? 0) - Self.unknownPenalty
        self.maxPieceLength = maxLen
    }

    /// Tokenize text into token IDs using Viterbi unigram decoding.
    ///
    /// Applies the standard SentencePiece normalization: replaces spaces
    /// with `\u{2581}` and prepends `\u{2581}` to the input.
    public func encode(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }

        // Normalize: prepend space marker, replace spaces with marker
        let normalized =
            String(Self.spaceMarker)
            + text.replacingOccurrences(
                of: " ", with: String(Self.spaceMarker))

        return viterbiDecode(normalized)
    }

    // MARK: - Viterbi Decoding

    /// Run Viterbi algorithm to find the highest-score segmentation.
    ///
    /// For each position in the string, finds the best-scoring
    /// vocabulary piece ending at that position.
    private func viterbiDecode(_ text: String) -> [Int] {
        let scalars = Array(text.unicodeScalars)
        let n = scalars.count
        guard n > 0 else { return [] }

        // bestScore[i] = best log-probability score for text[0..<i]
        // bestStep[i] = token IDs and start position for the step ending at i
        let negInf: Float = -.infinity
        var bestScore = [Float](repeating: negInf, count: n + 1)
        var bestStep = [DecodingStep?](repeating: nil, count: n + 1)
        bestScore[0] = 0

        // Build a string from scalars for substring matching
        // We work with Unicode scalar offsets for correctness
        for i in 0..<n {
            guard bestScore[i] > negInf else { continue }

            let maxLen = min(maxPieceLength, n - i)
            var hasSingleScalarPiece = false
            if maxLen > 0 {
                for length in 1...maxLen {
                    let end = i + length
                    // Build candidate substring from scalars
                    let candidate = String(String.UnicodeScalarView(scalars[i..<end]))

                    guard let pieceId = pieceToId[candidate] else { continue }
                    if length == 1 {
                        hasSingleScalarPiece = true
                    }
                    let piece = pieces[pieceId]

                    let pieceScore: Float
                    if piece.type == .userDefined {
                        pieceScore = 0.1 * Float(length - 1)
                    } else {
                        pieceScore = piece.score
                    }
                    let newScore = bestScore[i] + pieceScore
                    if newScore > bestScore[end] {
                        bestScore[end] = newScore
                        bestStep[end] = DecodingStep(tokenIds: [pieceId], start: i)
                    }
                }
            }

            guard !hasSingleScalarPiece, let unknownPieceId else { continue }

            let fallbackIds = fallbackTokenIds(for: scalars[i], unknownPieceId: unknownPieceId)
            let fallbackScore = bestScore[i] + unknownScore
            let end = i + 1
            if fallbackScore > bestScore[end] {
                bestScore[end] = fallbackScore
                bestStep[end] = DecodingStep(tokenIds: fallbackIds, start: i)
            }
        }

        // Backtrack to collect token IDs
        guard bestScore[n] > negInf else { return [] }

        var reversedSteps: [[Int]] = []
        var pos = n
        while pos > 0 {
            guard let step = bestStep[pos] else { return [] }
            reversedSteps.append(step.tokenIds)
            pos = step.start
        }

        let tokenIds = reversedSteps.reversed().flatMap { $0 }
        guard !byteFallbackEnabled, let unknownPieceId else { return tokenIds }

        return tokenIds.reduce(into: []) { mergedIds, tokenId in
            if tokenId != unknownPieceId || mergedIds.last != unknownPieceId {
                mergedIds.append(tokenId)
            }
        }
    }

    private func fallbackTokenIds(for scalar: Unicode.Scalar, unknownPieceId: Int) -> [Int] {
        guard byteFallbackEnabled else { return [unknownPieceId] }

        let bytes = Array(String(scalar).utf8)
        let byteIds = bytes.compactMap { bytePieceIds[$0] }
        if byteIds.count == bytes.count {
            return byteIds
        }

        return [unknownPieceId]
    }

    private static func byteValue(for piece: String) -> UInt8? {
        guard piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">") else {
            return nil
        }

        return UInt8(piece.dropFirst(3).dropLast(), radix: 16)
    }

    private struct DecodingStep {
        let tokenIds: [Int]
        let start: Int
    }
}
