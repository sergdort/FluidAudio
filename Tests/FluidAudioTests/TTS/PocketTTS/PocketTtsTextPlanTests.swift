@testable import FluidAudio
import Foundation
import XCTest

/// Tests for the fork-only text-plan and source-mapping path
/// (`makeTextPlan`) plus the normalization additions that feed it
/// (speakable-abbreviation expansion and trailing-quote preservation).
final class PocketTtsTextPlanTests: XCTestCase {
    private static let spaceMarker = "\u{2581}"

    // MARK: - Normalization additions (b42cec2f, still missing upstream)

    func testNormalizeTextExpandsIeToSpeakableText() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("Use i.e. when restating the rule")

        XCTAssertTrue(text.contains("that is"))
        XCTAssertFalse(text.contains("i.e."))
    }

    func testNormalizeTextExpandsEgToSpeakableText() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("Bring citrus, e.g. lemons and limes")

        XCTAssertTrue(text.contains("for example"))
        XCTAssertFalse(text.contains("e.g."))
    }

    func testNormalizeTextLeavesWorkingAbbreviationsUnchanged() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("Debate policy vs. precedent, etc.")

        XCTAssertTrue(text.contains("vs."))
        XCTAssertTrue(text.contains("etc."))
    }

    func testNormalizeTextPreservesQuotedSentenceEnding() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("\u{201C}This is a simple sentence.\u{201D}")
        XCTAssertEqual(text, "\"This is a simple sentence.\"")
    }

    func testNormalizeTextPreservesQuotedQuestionEnding() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("\u{201C}How can you be so tiresome?\u{201D}")
        XCTAssertEqual(text, "\"How can you be so tiresome?\"")
    }

    func testNormalizeTextDoesNotDoublePeriodOnQuotedSentence() {
        // A quoted sentence that already ends terminally inside the quote must
        // not gain a second period (the original b42cec2f regression).
        let (text, _) = PocketTtsSynthesizer.normalizeText("\u{201C}He walked away.\u{201D}")
        XCTAssertTrue(text.hasSuffix("\"He walked away.\""), "got: '\(text)'")
        XCTAssertFalse(text.contains(".\"."), "Must not append a second period after a terminal quote")
    }

    // MARK: - makeTextPlan (e0e03134)

    func testTextPlanNormalizedTextMatchesNormalizeTextOfSynthesisText() throws {
        // The plan's normalizedText must equal normalizeText applied to its
        // synthesisText (the exact string fed to tokenization at synthesis).
        let text = "Hello there. Goodbye now."
        let tokenizer = try makeCharacterTokenizer(for: [text])

        let plan = PocketTtsSynthesizer.makeTextPlan(text, tokenizer: tokenizer, maxTokens: 13)

        XCTAssertGreaterThan(plan.chunks.count, 1)
        for chunk in plan.chunks {
            XCTAssertEqual(
                chunk.normalizedText,
                PocketTtsSynthesizer.normalizeText(
                    chunk.synthesisText, isMidSentence: chunk.isMidSentence
                ).text
            )
        }
    }

    func testTextPlanSourceRangesExtractExactSourceText() throws {
        let text = "red blue red blue"
        let tokenizer = try makeCharacterTokenizer(for: [text])

        let plan = PocketTtsSynthesizer.makeTextPlan(text, tokenizer: tokenizer, maxTokens: 8)

        XCTAssertEqual(plan.chunks.map(\.sourceText), ["red", "blue", "red", "blue"])
        for chunk in plan.chunks {
            XCTAssertEqual(sourceSubstring(in: plan.originalText, range: chunk.sourceRange), chunk.sourceText)
        }
        XCTAssertNotEqual(plan.chunks[0].sourceRange, plan.chunks[2].sourceRange)
    }

    func testTextPlanKeepsAbbreviationSourceWhileNormalizedTextExpandsIt() throws {
        let text = "Use i.e. when restating the rule"
        let tokenizer = try makeCharacterTokenizer(for: [text])

        let plan = PocketTtsSynthesizer.makeTextPlan(text, tokenizer: tokenizer, maxTokens: 100)
        let chunk = try XCTUnwrap(plan.chunks.first)

        XCTAssertEqual(chunk.sourceText, text)
        XCTAssertTrue(chunk.normalizedText.contains("that is"))
        XCTAssertFalse(chunk.normalizedText.contains("i.e."))
        XCTAssertEqual(sourceSubstring(in: plan.originalText, range: chunk.sourceRange), text)
    }

    func testTextPlanKeepsClosingQuoteWithSentenceBoundary() throws {
        // The live (session) path absorbs a trailing closing quote into the
        // preceding boundary chunk, unlike the plain-string `chunkText`.
        let text = "\u{201C}Hello.\u{201D} \u{201C}Bye.\u{201D}"
        let tokenizer = try makeCharacterTokenizer(for: ["\"Hello.\"", "\"Bye.\""])

        let plan = PocketTtsSynthesizer.makeTextPlan(text, tokenizer: tokenizer, maxTokens: 10)

        XCTAssertEqual(plan.chunks.map(\.synthesisText), ["\"Hello.\"", "\"Bye.\""])
    }

    func testTextPlanPopulatesWordsWithSourceRanges() throws {
        let text = "Hello world"
        let tokenizer = try makeCharacterTokenizer(for: [text])

        let plan = PocketTtsSynthesizer.makeTextPlan(text, tokenizer: tokenizer, maxTokens: 100)
        let chunk = try XCTUnwrap(plan.chunks.first)

        XCTAssertEqual(chunk.words.map(\.sourceText), ["Hello", "world"])
        for word in chunk.words {
            XCTAssertEqual(sourceSubstring(in: plan.originalText, range: word.sourceRange), word.sourceText)
        }
    }

    func testTextPlanFlagsMidSentenceContinuationChunks() throws {
        // A single oversized sentence split at word boundaries must mark the
        // continuation chunk(s) mid-sentence so the synthesizer preserves
        // casing/punctuation (upstream #584 behaviour, threaded through the
        // source-mapped chunker).
        let text = "alpha beta gamma delta epsilon"
        let tokenizer = try makeCharacterTokenizer(for: [text])

        let plan = PocketTtsSynthesizer.makeTextPlan(text, tokenizer: tokenizer, maxTokens: 12)

        XCTAssertGreaterThan(plan.chunks.count, 1)
        XCTAssertEqual(plan.chunks.first?.isMidSentence, false)
        XCTAssertTrue(plan.chunks.dropFirst().allSatisfy { $0.isMidSentence })
    }

    // MARK: - Helpers

    private func sourceSubstring(in text: String, range: PocketTtsSourceRange) -> String? {
        guard
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

    private func makeCharacterTokenizer(for texts: [String]) throws -> SentencePieceTokenizer {
        let characters = Set((texts.joined() + Self.spaceMarker).map(String.init))
        let pieces = characters.sorted().map { makePieceMessage(string: $0, score: 0) }
        return try SentencePieceTokenizer(modelData: wrapInModelProto(pieces: pieces))
    }

    private func makeVarint(_ value: UInt64) -> [UInt8] {
        var result: [UInt8] = []
        var currentValue = value
        while currentValue > 0x7F {
            result.append(UInt8(currentValue & 0x7F) | 0x80)
            currentValue >>= 7
        }
        result.append(UInt8(currentValue))
        return result
    }

    private func makeTag(fieldNumber: Int, wireType: Int) -> [UInt8] {
        makeVarint(UInt64((fieldNumber << 3) | wireType))
    }

    private func makeFloat32Bytes(_ value: Float) -> [UInt8] {
        var currentValue = value
        return withUnsafeBytes(of: &currentValue) { Array($0) }
    }

    private func makePieceMessage(string: String, score: Float) -> [UInt8] {
        var body: [UInt8] = []
        let stringBytes = Array(string.utf8)
        body.append(contentsOf: makeTag(fieldNumber: 1, wireType: 2))
        body.append(contentsOf: makeVarint(UInt64(stringBytes.count)))
        body.append(contentsOf: stringBytes)
        body.append(contentsOf: makeTag(fieldNumber: 2, wireType: 5))
        body.append(contentsOf: makeFloat32Bytes(score))
        return body
    }

    private func wrapInModelProto(pieces: [[UInt8]]) -> Data {
        var data: [UInt8] = []
        for piece in pieces {
            data.append(contentsOf: makeTag(fieldNumber: 1, wireType: 2))
            data.append(contentsOf: makeVarint(UInt64(piece.count)))
            data.append(contentsOf: piece)
        }
        return Data(data)
    }
}
