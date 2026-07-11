import Foundation

/// Text-plan and estimated-highlight support for PocketTTS.
///
/// This is a fork-only extension (Oratio). It produces a `TextPlan` that maps
/// synthesized audio chunks back to their exact source-text ranges (UTF-16
/// offsets into the original enqueued string), plus per-word source ranges used
/// to estimate word-level highlight timing without an alignment model.
///
/// It runs alongside upstream's `chunkTextWithMetadata` path: it carries the
/// same `isMidSentence` prosody information (issue #584) so the session can
/// normalize mid-sentence continuations correctly, while additionally tracking
/// source ranges that upstream's plain-string chunker discards.
extension PocketTtsSynthesizer {

    // MARK: - Public Plan Types

    public struct TextPlan: Sendable, Equatable {
        public let originalText: String
        public let chunks: [PlanChunk]

        public init(originalText: String, chunks: [PlanChunk]) {
            self.originalText = originalText
            self.chunks = chunks
        }
    }

    /// A single synthesizable chunk of a text plan.
    ///
    /// Named `PlanChunk` to avoid colliding with upstream's
    /// `PocketTtsSynthesizer.TextChunk` (`text` + `isMidSentence`), which the
    /// plain-string chunker (`chunkTextWithMetadata`) uses. `PocketTtsTextChunk`
    /// aliases this richer type for consumers.
    public struct PlanChunk: Sendable, Equatable, Identifiable {
        public let id: Int
        /// UTF-16 offsets into `TextPlan.originalText`.
        public let sourceRange: SourceRange
        /// Exact source substring from `TextPlan.originalText`.
        public let sourceText: String
        /// Chunk text used by the PocketTTS synthesis path before `normalizeText`.
        public let synthesisText: String
        /// Exact normalized text passed to tokenization/model input.
        public let normalizedText: String
        /// Source words in this chunk, using UTF-16 offsets into `TextPlan.originalText`.
        public let words: [TextPlanWord]
        /// True when this chunk is a continuation of a sentence (a clause- or
        /// word-boundary split), so casing/terminal punctuation are preserved.
        public let isMidSentence: Bool

        public init(
            id: Int,
            sourceRange: SourceRange,
            sourceText: String,
            synthesisText: String,
            normalizedText: String,
            words: [TextPlanWord] = [],
            isMidSentence: Bool = false
        ) {
            self.id = id
            self.sourceRange = sourceRange
            self.sourceText = sourceText
            self.synthesisText = synthesisText
            self.normalizedText = normalizedText
            self.words = words
            self.isMidSentence = isMidSentence
        }
    }

    public struct TextPlanWord: Sendable, Equatable, Identifiable {
        public let id: Int
        /// UTF-16 offsets into `TextPlan.originalText`.
        public let sourceRange: SourceRange
        /// Exact source substring from `TextPlan.originalText`.
        public let sourceText: String
        /// Normalized word text used for timing estimation.
        public let normalizedText: String

        public init(id: Int, sourceRange: SourceRange, sourceText: String, normalizedText: String) {
            self.id = id
            self.sourceRange = sourceRange
            self.sourceText = sourceText
            self.normalizedText = normalizedText
        }
    }

    public struct HighlightSpan: Sendable, Equatable, Identifiable {
        public enum Style: Sendable, Equatable {
            case word
            case reading
        }

        public let id: Int
        public let style: Style
        public let sourceRange: SourceRange
        public let startTime: TimeInterval
        public let endTime: TimeInterval

        public init(id: Int, style: Style, sourceRange: SourceRange, startTime: TimeInterval, endTime: TimeInterval) {
            self.id = id
            self.style = style
            self.sourceRange = sourceRange
            self.startTime = startTime
            self.endTime = endTime
        }
    }

    public struct SourceRange: Sendable, Equatable {
        public let lowerBound: Int
        public let upperBound: Int

        public init(lowerBound: Int, upperBound: Int) {
            self.lowerBound = lowerBound
            self.upperBound = upperBound
        }
    }

    public enum SessionEvent: Sendable {
        case utterancePlanned(utteranceIndex: Int, plan: TextPlan)
        case audioFrame(AudioFrame)
        case chunkHighlights(utteranceIndex: Int, chunkIndex: Int, audioDuration: TimeInterval, spans: [HighlightSpan])
    }

    // MARK: - Plan Construction

    /// Build a `TextPlan` for `text`, mapping each synthesizable chunk (and its
    /// words) back to exact UTF-16 source ranges.
    ///
    /// Chunk boundaries mirror `chunkTextWithMetadata`, including the
    /// `isMidSentence` prosody flag (#584); the `language` parameter selects the
    /// normalization used for each chunk's `normalizedText`.
    public static func makeTextPlan(
        _ text: String,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int = PocketTtsConstants.maxTokensPerChunk,
        language: PocketTtsLanguage = .english
    ) -> TextPlan {
        let mappedText = makeMappedCanonicalText(from: text)
        let mappedChunks = chunkMappedText(mappedText, tokenizer: tokenizer, maxTokens: maxTokens)
        let chunks = mappedChunks.enumerated().map { index, mappedChunk -> PlanChunk in
            let sourceText = sourceSubstring(in: text, range: mappedChunk.text.sourceRange) ?? ""
            let words = splitMappedWords(mappedChunk.text).enumerated().map { wordIndex, word in
                TextPlanWord(
                    id: wordIndex,
                    sourceRange: word.sourceRange,
                    sourceText: sourceSubstring(in: text, range: word.sourceRange) ?? "",
                    normalizedText: word.text
                )
            }
            return PlanChunk(
                id: index,
                sourceRange: mappedChunk.text.sourceRange,
                sourceText: sourceText,
                synthesisText: mappedChunk.text.text,
                normalizedText: normalizeText(
                    mappedChunk.text.text,
                    isMidSentence: mappedChunk.isMidSentence,
                    language: language
                ).text,
                words: words,
                isMidSentence: mappedChunk.isMidSentence
            )
        }

        return TextPlan(originalText: text, chunks: chunks)
    }

    // MARK: - Estimated Highlights

    /// Estimate word-level highlight spans for a chunk by distributing the
    /// measured audio duration across words weighted by their non-whitespace
    /// character count. Emits a `.word` span and a cumulative `.reading` span
    /// per word. Returns `[]` when there is no audio or no words.
    public static func estimatedHighlightSpans(
        for chunk: PlanChunk,
        audioDuration: TimeInterval
    ) -> [HighlightSpan] {
        guard audioDuration > 0, chunk.words.isEmpty == false else {
            return []
        }

        let weights = chunk.words.map { word in
            max(1.0, Double(word.normalizedText.filter { $0.isWhitespace == false }.count))
        }
        let totalWeight = weights.reduce(0, +)
        guard totalWeight > 0 else { return [] }

        var spans: [HighlightSpan] = []
        spans.reserveCapacity(chunk.words.count * 2)

        var elapsed: TimeInterval = 0
        for (index, word) in chunk.words.enumerated() {
            let startTime = elapsed
            let duration =
                index == chunk.words.indices.last
                ? audioDuration - elapsed : audioDuration * weights[index] / totalWeight
            let endTime = min(audioDuration, startTime + max(0, duration))

            spans.append(
                HighlightSpan(
                    id: spans.count,
                    style: .word,
                    sourceRange: word.sourceRange,
                    startTime: startTime,
                    endTime: endTime
                ))
            spans.append(
                HighlightSpan(
                    id: spans.count,
                    style: .reading,
                    sourceRange: SourceRange(
                        lowerBound: chunk.words[0].sourceRange.lowerBound,
                        upperBound: word.sourceRange.upperBound
                    ),
                    startTime: startTime,
                    endTime: endTime
                ))

            elapsed = endTime
        }

        return spans
    }

    // MARK: - Source-Mapped Chunking

    private struct MappedChunk {
        let text: MappedText
        let isMidSentence: Bool
    }

    private static func chunkMappedText(
        _ text: MappedText,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int
    ) -> [MappedChunk] {
        guard text.text.isEmpty == false else {
            return []
        }

        let tokenCount = tokenizer.encode(text.text).count
        if tokenCount <= maxTokens {
            return [MappedChunk(text: text, isMidSentence: false)]
        }

        let sentences = splitMappedSentences(text)

        // Split oversized sentences at clause/word boundaries, tagging the
        // continuation pieces as mid-sentence so the synthesizer preserves
        // their casing/punctuation (#584).
        var pieces: [MappedChunk] = []
        for sentence in sentences {
            let sentenceTokens = tokenizer.encode(sentence.text).count
            if sentenceTokens <= maxTokens {
                pieces.append(MappedChunk(text: sentence, isMidSentence: false))
            } else {
                let subPieces = splitMappedOversizedSentence(sentence, tokenizer: tokenizer, maxTokens: maxTokens)
                for (subIndex, subPiece) in subPieces.enumerated() {
                    pieces.append(MappedChunk(text: subPiece, isMidSentence: subIndex > 0))
                }
            }
        }

        // Group pieces into chunks that fit, merging only pieces whose
        // mid-sentence flags match so boundary cues are not lost.
        var chunks: [MappedChunk] = []
        var current: MappedChunk?

        for piece in pieces {
            guard let existing = current else {
                current = piece
                continue
            }

            if existing.isMidSentence != piece.isMidSentence {
                chunks.append(existing)
                current = piece
                continue
            }

            let candidate = existing.text.joined(with: piece.text)
            if tokenizer.encode(candidate.text).count <= maxTokens {
                current = MappedChunk(text: candidate, isMidSentence: existing.isMidSentence)
            } else {
                chunks.append(existing)
                current = piece
            }
        }

        if let current {
            chunks.append(current)
        }

        return chunks.isEmpty ? [MappedChunk(text: text, isMidSentence: false)] : chunks
    }

    // MARK: - Mapped Text Primitives

    private struct MappedCharacter {
        let character: Character
        let sourceRange: SourceRange
    }

    private struct MappedText {
        let characters: [MappedCharacter]

        var text: String {
            String(characters.map(\.character))
        }

        var sourceRange: SourceRange {
            guard let first = characters.first, let last = characters.last else {
                return SourceRange(lowerBound: 0, upperBound: 0)
            }

            return SourceRange(
                lowerBound: first.sourceRange.lowerBound,
                upperBound: last.sourceRange.upperBound
            )
        }

        func trimmed() -> MappedText? {
            var lowerBound = characters.startIndex
            var upperBound = characters.endIndex

            while lowerBound < upperBound, characters[lowerBound].character.isWhitespace {
                lowerBound = characters.index(after: lowerBound)
            }

            while upperBound > lowerBound, characters[characters.index(before: upperBound)].character.isWhitespace {
                upperBound = characters.index(before: upperBound)
            }

            guard lowerBound < upperBound else {
                return nil
            }

            return MappedText(characters: Array(characters[lowerBound..<upperBound]))
        }

        func joined(with next: MappedText) -> MappedText {
            var joinedCharacters = characters
            if let previous = characters.last {
                joinedCharacters.append(
                    MappedCharacter(
                        character: " ",
                        sourceRange: SourceRange(
                            lowerBound: previous.sourceRange.upperBound,
                            upperBound: previous.sourceRange.upperBound
                        )
                    ))
            }
            joinedCharacters.append(contentsOf: next.characters)
            return MappedText(characters: joinedCharacters)
        }
    }

    private static func makeMappedCanonicalText(from text: String) -> MappedText {
        var characters: [MappedCharacter] = []
        var index = text.startIndex

        while index < text.endIndex {
            let nextIndex = text.index(after: index)
            let character = text[index]
            let mappedCharacter: Character
            switch character {
            case "‘", "’":
                mappedCharacter = "'"
            case "“", "”":
                mappedCharacter = "\""
            default:
                mappedCharacter = character
            }

            characters.append(
                MappedCharacter(
                    character: mappedCharacter,
                    sourceRange: SourceRange(
                        lowerBound: text.utf16.distance(
                            from: text.utf16.startIndex, to: index.samePosition(in: text.utf16)!),
                        upperBound: text.utf16.distance(
                            from: text.utf16.startIndex, to: nextIndex.samePosition(in: text.utf16)!)
                    )
                ))

            index = nextIndex
        }

        return MappedText(characters: characters).trimmed() ?? MappedText(characters: [])
    }

    private static func sourceSubstring(in text: String, range: SourceRange) -> String? {
        guard range.lowerBound >= 0,
            range.upperBound >= range.lowerBound,
            let lowerUTF16 = text.utf16.index(
                text.utf16.startIndex, offsetBy: range.lowerBound, limitedBy: text.utf16.endIndex),
            let upperUTF16 = text.utf16.index(
                text.utf16.startIndex, offsetBy: range.upperBound, limitedBy: text.utf16.endIndex),
            let lower = String.Index(lowerUTF16, within: text),
            let upper = String.Index(upperUTF16, within: text)
        else {
            return nil
        }

        return String(text[lower..<upper])
    }

    private static func splitMappedOversizedSentence(
        _ text: MappedText,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int
    ) -> [MappedText] {
        let clauseParts = splitMappedAtClauseBoundaries(text)
        var result: [MappedText] = []
        var currentPart: MappedText?

        for part in clauseParts {
            let candidate = currentPart.map { $0.joined(with: part) } ?? part
            let candidateTokens = tokenizer.encode(candidate.text).count

            if candidateTokens <= maxTokens {
                currentPart = candidate
            } else {
                if let currentPart {
                    result.append(currentPart)
                }
                if tokenizer.encode(part.text).count > maxTokens {
                    result.append(
                        contentsOf: splitMappedAtWordBoundaries(part, tokenizer: tokenizer, maxTokens: maxTokens))
                    currentPart = nil
                } else {
                    currentPart = part
                }
            }
        }

        if let currentPart {
            result.append(currentPart)
        }

        return result.isEmpty ? [text] : result
    }

    private static func splitMappedAtClauseBoundaries(_ text: MappedText) -> [MappedText] {
        let clauseBreaks: Set<Character> = [",", ";", ":"]
        var parts: [MappedText] = []
        var current: [MappedCharacter] = []
        let characters = text.characters

        var index = 0
        while index < characters.count {
            let mapped = characters[index]
            current.append(mapped)

            guard clauseBreaks.contains(mapped.character) else {
                index += 1
                continue
            }

            if mapped.character == "," {
                let previousIsDigit = index > 0 && characters[index - 1].character.isNumber
                let nextIsDigit = index + 1 < characters.count && characters[index + 1].character.isNumber
                if previousIsDigit, nextIsDigit {
                    index += 1
                    continue
                }
            }

            index += 1
            while index < characters.count, characters[index].character == "\"" {
                current.append(characters[index])
                index += 1
            }

            if let trimmed = MappedText(characters: current).trimmed() {
                parts.append(trimmed)
            }
            current = []
        }

        if let trimmed = MappedText(characters: current).trimmed() {
            parts.append(trimmed)
        }

        return parts
    }

    private static func splitMappedAtWordBoundaries(
        _ text: MappedText,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int
    ) -> [MappedText] {
        let words = splitMappedWords(text)
        guard words.count > 1 else { return [text] }

        var chunks: [MappedText] = []
        var currentWords: [MappedText] = []

        for word in words {
            let candidate = joinMappedWords(currentWords + [word])
            let tokens = tokenizer.encode(candidate.text).count

            if tokens > maxTokens, !currentWords.isEmpty {
                chunks.append(joinMappedWords(currentWords))
                currentWords = [word]
            } else {
                currentWords.append(word)
            }
        }

        if !currentWords.isEmpty {
            chunks.append(joinMappedWords(currentWords))
        }

        return chunks
    }

    private static func splitMappedWords(_ text: MappedText) -> [MappedText] {
        var words: [MappedText] = []
        var current: [MappedCharacter] = []

        for character in text.characters {
            if character.character == " " {
                if current.isEmpty == false {
                    words.append(MappedText(characters: current))
                    current = []
                }
            } else {
                current.append(character)
            }
        }

        if current.isEmpty == false {
            words.append(MappedText(characters: current))
        }

        return words
    }

    private static func joinMappedWords(_ words: [MappedText]) -> MappedText {
        guard var result = words.first else {
            return MappedText(characters: [])
        }

        for word in words.dropFirst() {
            result = result.joined(with: word)
        }

        return result
    }

    private static func splitMappedSentences(_ text: MappedText) -> [MappedText] {
        var sentences: [MappedText] = []
        var current: [MappedCharacter] = []
        let characters = text.characters

        var index = 0
        while index < characters.count {
            let mapped = characters[index]
            current.append(mapped)

            guard ".!?".contains(mapped.character) else {
                index += 1
                continue
            }

            if mapped.character == "." {
                let currentText = MappedText(characters: current).text.trimmingCharacters(in: .whitespaces)
                let withoutPeriod = String(currentText.dropLast())
                let lastWord = withoutPeriod.split(separator: " ").last.map(String.init) ?? withoutPeriod

                if abbreviations.contains(lastWord.lowercased()) {
                    index += 1
                    continue
                }

                if lastWord.count == 1, lastWord.first?.isUppercase == true {
                    index += 1
                    continue
                }

                if index + 1 < characters.count, characters[index + 1].character.isNumber {
                    index += 1
                    continue
                }
            }

            index += 1
            while index < characters.count, characters[index].character == "\"" {
                current.append(characters[index])
                index += 1
            }

            if let trimmed = MappedText(characters: current).trimmed() {
                sentences.append(trimmed)
            }
            current = []
        }

        if let trimmed = MappedText(characters: current).trimmed() {
            sentences.append(trimmed)
        }

        return sentences
    }
}

public typealias PocketTtsTextPlan = PocketTtsSynthesizer.TextPlan
public typealias PocketTtsTextChunk = PocketTtsSynthesizer.PlanChunk
public typealias PocketTtsTextPlanWord = PocketTtsSynthesizer.TextPlanWord
public typealias PocketTtsSourceRange = PocketTtsSynthesizer.SourceRange
public typealias PocketTtsHighlightSpan = PocketTtsSynthesizer.HighlightSpan
public typealias PocketTtsSessionEvent = PocketTtsSynthesizer.SessionEvent
