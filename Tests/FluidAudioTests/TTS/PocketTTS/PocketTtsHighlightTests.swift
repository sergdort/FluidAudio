@testable import FluidAudio
import Foundation
import XCTest

final class PocketTtsHighlightTests: XCTestCase {
    func testEstimatedHighlightSpansProducesWordAndReadingTracks() {
        let chunk = PocketTtsSynthesizer.TextChunk(
            id: 0,
            sourceRange: .init(lowerBound: 0, upperBound: 11),
            sourceText: "Hello world",
            synthesisText: "Hello world",
            normalizedText: "Hello world",
            words: [
                .init(id: 0, sourceRange: .init(lowerBound: 0, upperBound: 5), sourceText: "Hello", normalizedText: "Hello"),
                .init(id: 1, sourceRange: .init(lowerBound: 6, upperBound: 11), sourceText: "world", normalizedText: "world"),
            ]
        )

        let spans = PocketTtsSynthesizer.estimatedHighlightSpans(for: chunk, audioDuration: 2.0)

        XCTAssertEqual(spans, [
            .init(id: 0, style: .word, sourceRange: .init(lowerBound: 0, upperBound: 5), startTime: 0.0, endTime: 1.0),
            .init(id: 1, style: .reading, sourceRange: .init(lowerBound: 0, upperBound: 5), startTime: 0.0, endTime: 1.0),
            .init(id: 2, style: .word, sourceRange: .init(lowerBound: 6, upperBound: 11), startTime: 1.0, endTime: 2.0),
            .init(id: 3, style: .reading, sourceRange: .init(lowerBound: 0, upperBound: 11), startTime: 1.0, endTime: 2.0),
        ])
    }

    func testEstimatedHighlightSpansReturnsEmptyWithoutWords() {
        let chunk = PocketTtsSynthesizer.TextChunk(
            id: 0,
            sourceRange: .init(lowerBound: 0, upperBound: 5),
            sourceText: "Hello",
            synthesisText: "Hello",
            normalizedText: "Hello"
        )

        XCTAssertEqual(PocketTtsSynthesizer.estimatedHighlightSpans(for: chunk, audioDuration: 1.0), [])
    }
}
