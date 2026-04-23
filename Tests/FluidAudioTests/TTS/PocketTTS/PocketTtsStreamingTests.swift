import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsStreamingTests: XCTestCase {

    private static let spaceMarker = "\u{2581}"

    // MARK: - AudioFrame Tests

    func testAudioFrameProperties() {
        let samples: [Float] = Array(repeating: 0.5, count: PocketTtsConstants.samplesPerFrame)
        let frame = PocketTtsSynthesizer.AudioFrame(
            samples: samples,
            frameIndex: 3,
            chunkIndex: 1,
            chunkCount: 4,
            utteranceIndex: nil
        )

        XCTAssertEqual(frame.samples.count, PocketTtsConstants.samplesPerFrame)
        XCTAssertEqual(frame.frameIndex, 3)
        XCTAssertEqual(frame.chunkIndex, 1)
        XCTAssertEqual(frame.chunkCount, 4)
        XCTAssertNil(frame.utteranceIndex)
    }

    func testAudioFrameIsSendable() {
        // Verify AudioFrame can be sent across concurrency boundaries
        let frame = PocketTtsSynthesizer.AudioFrame(
            samples: [1.0, 2.0, 3.0],
            frameIndex: 0,
            chunkIndex: 0,
            chunkCount: 1,
            utteranceIndex: nil
        )

        let expectation = expectation(description: "Frame sent across tasks")
        Task {
            let _: PocketTtsSynthesizer.AudioFrame = frame
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)
    }

    // MARK: - PocketTtsManager Guard Tests

    func testSynthesizeStreamingFailsWithoutInitialization() async {
        let manager = PocketTtsManager()

        do {
            _ = try await manager.synthesizeStreaming(text: "Hello")
            XCTFail("Expected error when not initialized")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testSynthesizeStreamingWithVoiceDataFailsWithoutInitialization() async {
        let manager = PocketTtsManager()
        let fakeVoiceData = PocketTtsVoiceData(audioPrompt: [], promptLength: 0)

        do {
            _ = try await manager.synthesizeStreaming(text: "Hello", voiceData: fakeVoiceData)
            XCTFail("Expected error when not initialized")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - Text Normalization (used by streaming pipeline)

    func testNormalizeTextAddsTerminalPunctuation() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("Hello world")
        XCTAssertTrue(text.hasSuffix("."), "Should add period when no terminal punctuation")
    }

    func testNormalizeTextPreservesExistingPunctuation() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("Hello world!")
        XCTAssertTrue(text.hasSuffix("!"), "Should preserve existing punctuation")
        XCTAssertFalse(text.hasSuffix("!."), "Should not add extra period")
    }

    func testNormalizeTextPreservesQuotedSentenceEnding() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("“This is a simple sentence.”")
        XCTAssertEqual(text, "\"This is a simple sentence.\"")
    }

    func testNormalizeTextPreservesQuotedQuestionEnding() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("“How can you be so tiresome?”")
        XCTAssertEqual(text, "\"How can you be so tiresome?\"")
    }

    func testNormalizeTextCanonicalizesSmartQuotesInDialogueSample() {
        let sample = "“My dear Mr. Bennet,” replied his wife, “how can you be so tiresome? You must know that I am thinking of his marrying one of them.”"
        let (text, _) = PocketTtsSynthesizer.normalizeText(sample)

        XCTAssertFalse(text.contains("“"))
        XCTAssertFalse(text.contains("”"))
        XCTAssertTrue(text.contains("\"My dear Mr. Bennet,"))
        XCTAssertTrue(text.hasSuffix(".\""))
    }

    func testNormalizeTextCanonicalizesSmartApostrophesInContractions() {
        let sample = "I’ve been turning over in my mind ever since"
        let (text, _) = PocketTtsSynthesizer.normalizeText(sample)

        XCTAssertTrue(text.contains("I've been"))
        XCTAssertFalse(text.contains("I’ve"))
    }

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

    func testNormalizeTextCapitalizesFirstLetter() {
        let (text, _) = PocketTtsSynthesizer.normalizeText("hello")
        XCTAssertTrue(text.contains("H"), "Should capitalize first letter")
    }

    func testNormalizeTextShortTextPadding() {
        // Short text (< 5 words) gets padding
        let (text, frames) = PocketTtsSynthesizer.normalizeText("Hi")
        XCTAssertTrue(text.hasPrefix(" "), "Short text should be padded")
        XCTAssertEqual(frames, PocketTtsConstants.shortTextPadFrames)
    }

    func testNormalizeTextLongTextNoExtraPadding() {
        let (_, frames) = PocketTtsSynthesizer.normalizeText(
            "This is a longer sentence with more than five words in it")
        XCTAssertEqual(frames, PocketTtsConstants.longTextExtraFrames)
    }

    func testChunkTextKeepsClosingQuoteWithSentenceBoundary() throws {
        let tokenizer = try makeCharacterTokenizer(for: ["\"Hello.\"", "\"Bye.\""])

        let chunks = PocketTtsSynthesizer.chunkText(
            "“Hello.” “Bye.”",
            tokenizer: tokenizer,
            maxTokens: 10
        )

        XCTAssertEqual(chunks, ["\"Hello.\"", "\"Bye.\""])
    }

    func testChunkTextKeepsClosingQuoteWithClauseBoundary() throws {
        let tokenizer = try makeCharacterTokenizer(for: ["\"Hello,\" she said"])

        let chunks = PocketTtsSynthesizer.chunkText(
            "“Hello,” she said",
            tokenizer: tokenizer,
            maxTokens: 9
        )

        XCTAssertEqual(chunks, ["\"Hello,\"", "she said"])
    }

    func testChunkTextCanonicalizesSmartApostrophesBeforeTokenization() throws {
        let tokenizer = try makeCharacterTokenizer(for: ["I've been turning over in my mind ever since"])

        let chunks = PocketTtsSynthesizer.chunkText(
            "I’ve been turning over in my mind ever since",
            tokenizer: tokenizer,
            maxTokens: 100
        )

        XCTAssertEqual(chunks, ["I've been turning over in my mind ever since"])
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
