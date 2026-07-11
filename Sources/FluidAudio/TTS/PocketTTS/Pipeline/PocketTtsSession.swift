@preconcurrency import CoreML
import Foundation

/// A persistent TTS session that keeps the voice KV cache warm across utterances.
///
/// Creating a session performs the expensive voice prefill once (~125 tokens),
/// then each enqueued utterance only pays the text prefill cost. Mimi decoder
/// state persists across utterances for seamless audio continuity.
public actor PocketTtsSession {

    private static let logger = AppLogger(category: "PocketTtsSession")

    // MARK: - Public Interface

    /// Stream of text plans, generated audio frames, and estimated highlights.
    ///
    /// Each utterance emits `.utterancePlanned` before any `.audioFrame` for
    /// that utterance, allowing clients to map audio chunk boundaries back to
    /// source text without delaying audio playback. `.chunkHighlights` is
    /// emitted after a chunk's audio completes.
    ///
    /// This is the stream Oratio consumes; audio is delivered here as
    /// `.audioFrame` rather than on the legacy `frames` stream (see below).
    public nonisolated let events: AsyncThrowingStream<PocketTtsSynthesizer.SessionEvent, Error>

    /// Stream of generated audio frames (80ms / 1920 samples at 24kHz each).
    ///
    /// Retained for API compatibility. Audio is now delivered via `events`
    /// (`.audioFrame`); this stream only completes/faults alongside `events`
    /// and does not carry sample buffers, so an `events`-only consumer never
    /// accumulates an unbounded unused frame backlog.
    public nonisolated let frames: AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error>

    /// Enqueue text for synthesis.
    ///
    /// Non-async and safe to call from any isolation context. Text is chunked
    /// internally if it exceeds the per-chunk token limit. Can be called
    /// multiple times to stream text as it arrives.
    public nonisolated func enqueue(_ text: String) {
        textContinuation.yield(text)
    }

    /// Signal that no more text will be enqueued.
    ///
    /// The session will finish generating all previously enqueued text,
    /// then complete the `frames` stream.
    public nonisolated func finish() {
        textContinuation.finish()
    }

    /// Cancel ongoing generation and finish the frames stream.
    ///
    /// Awaits until the generation task has fully stopped — after this returns,
    /// no more CoreML predictions are running and the Neural Engine is free.
    public func cancel() async {
        generationTask?.cancel()
        textContinuation.finish()
        await generationTask?.value
    }

    // MARK: - Internal State

    private nonisolated let textContinuation: AsyncStream<String>.Continuation
    private let textStream: AsyncStream<String>
    private let eventContinuation: AsyncThrowingStream<PocketTtsSynthesizer.SessionEvent, Error>.Continuation
    private let frameContinuation: AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error>.Continuation
    private var generationTask: Task<Void, Never>?

    // Models (IO placements: `.gpu` / `.ane`). All `nil` for `.aneState`
    // sessions, which run on the multifunction state pipeline instead.
    private let condModel: MLModel?
    private let condPrefillModel: MLModel?
    private let useCondPrefill: Bool
    private let stepModel: MLModel?
    private let flowModel: MLModel?
    private let mimiModel: MLModel
    private let condLayerKeys: PocketTtsLayerKeys?
    private let condPrefillLayerKeys: PocketTtsLayerKeys?
    private let flowlmLayerKeys: PocketTtsLayerKeys?
    private let mimiKeys: PocketTtsMimiKeys

    // Persistent state (IO placements). `nil` for `.aneState`.
    private let voiceKVSnapshot: PocketTtsSynthesizer.KVCacheState?
    private let constants: PocketTtsConstantsBundle
    private let bosEmb: MLMultiArray
    private let temperature: Float
    private let language: PocketTtsLanguage
    private let maxTokensPerChunk: Int
    private var mimiState: PocketTtsSynthesizer.MimiState
    private var rng: SeededRNG

    // Stateful (`.aneState`) path — mobius Trial 23 MLState pipeline.
    // `stateEngineRef` / `stateVoiceSnapshotRef` are lazily created on the
    // first chunk and stored type-erased because `PocketTtsStateEngine` (and
    // its snapshot struct) are only available on macOS 15+/iOS 18+ while
    // this actor compiles for macOS 14; use sites downcast under #available.
    private let stateModels: PocketTtsStateModels?
    private let stateVoiceData: PocketTtsVoiceData?
    private var stateEngineRef: AnyObject?
    private var stateVoiceSnapshotRef: Any?

    // MARK: - Initialization

    /// Create a session with pre-computed voice KV cache.
    ///
    /// This initializer is internal — use `PocketTtsManager.makeSession()` instead.
    init(
        voiceKVSnapshot: PocketTtsSynthesizer.KVCacheState,
        mimiState: PocketTtsSynthesizer.MimiState,
        constants: PocketTtsConstantsBundle,
        condModel: MLModel,
        condPrefillModel: MLModel,
        useCondPrefill: Bool,
        stepModel: MLModel,
        flowModel: MLModel,
        mimiModel: MLModel,
        condLayerKeys: PocketTtsLayerKeys,
        condPrefillLayerKeys: PocketTtsLayerKeys?,
        flowlmLayerKeys: PocketTtsLayerKeys,
        mimiKeys: PocketTtsMimiKeys,
        bosEmb: MLMultiArray,
        temperature: Float,
        seed: UInt64,
        language: PocketTtsLanguage = .english,
        maxTokensPerChunk: Int = PocketTtsConstants.maxTokensPerChunk
    ) {
        self.voiceKVSnapshot = voiceKVSnapshot
        self.mimiState = mimiState
        self.constants = constants
        self.condModel = condModel
        self.condPrefillModel = condPrefillModel
        self.useCondPrefill = useCondPrefill
        self.stepModel = stepModel
        self.flowModel = flowModel
        self.mimiModel = mimiModel
        self.condLayerKeys = condLayerKeys
        self.condPrefillLayerKeys = condPrefillLayerKeys
        self.flowlmLayerKeys = flowlmLayerKeys
        self.mimiKeys = mimiKeys
        self.bosEmb = bosEmb
        self.temperature = temperature
        self.language = language
        self.maxTokensPerChunk = maxTokensPerChunk
        self.rng = SeededRNG(seed: seed)
        self.stateModels = nil
        self.stateVoiceData = nil

        // Text queue channel
        let (textStream, textContinuation) = AsyncStream.makeStream(of: String.self)
        self.textStream = textStream
        self.textContinuation = textContinuation

        // Event and frame output streams
        let (events, eventContinuation) = AsyncThrowingStream.makeStream(
            of: PocketTtsSynthesizer.SessionEvent.self
        )
        let (frames, frameContinuation) = AsyncThrowingStream.makeStream(
            of: PocketTtsSynthesizer.AudioFrame.self
        )
        self.events = events
        self.eventContinuation = eventContinuation
        self.frames = frames
        self.frameContinuation = frameContinuation
    }

    /// Create a stateful (`.aneState`) session over the Trial 23 MLState
    /// multifunction pipeline. The voice KV is injected straight into the
    /// shared `MLState` per chunk (fp16 snapshot write = utterance reset),
    /// so none of the IO models/keys are needed.
    ///
    /// This initializer is internal — use `PocketTtsManager.makeSession()`
    /// with a `PocketTtsModelStore` configured for `.aneState`.
    init(
        stateModels: PocketTtsStateModels,
        voiceData: PocketTtsVoiceData,
        mimiState: PocketTtsSynthesizer.MimiState,
        constants: PocketTtsConstantsBundle,
        mimiModel: MLModel,
        mimiKeys: PocketTtsMimiKeys,
        bosEmb: MLMultiArray,
        temperature: Float,
        seed: UInt64,
        language: PocketTtsLanguage = .english,
        maxTokensPerChunk: Int = PocketTtsConstants.maxTokensPerChunk
    ) {
        self.voiceKVSnapshot = nil
        self.mimiState = mimiState
        self.constants = constants
        self.condModel = nil
        self.condPrefillModel = nil
        self.useCondPrefill = false
        self.stepModel = nil
        self.flowModel = nil
        self.mimiModel = mimiModel
        self.condLayerKeys = nil
        self.condPrefillLayerKeys = nil
        self.flowlmLayerKeys = nil
        self.mimiKeys = mimiKeys
        self.bosEmb = bosEmb
        self.temperature = temperature
        self.language = language
        self.maxTokensPerChunk = maxTokensPerChunk
        self.rng = SeededRNG(seed: seed)
        self.stateModels = stateModels
        self.stateVoiceData = voiceData

        // Text queue channel
        let (textStream, textContinuation) = AsyncStream.makeStream(of: String.self)
        self.textStream = textStream
        self.textContinuation = textContinuation

        // Event and frame output streams
        let (events, eventContinuation) = AsyncThrowingStream.makeStream(
            of: PocketTtsSynthesizer.SessionEvent.self
        )
        let (frames, frameContinuation) = AsyncThrowingStream.makeStream(
            of: PocketTtsSynthesizer.AudioFrame.self
        )
        self.events = events
        self.eventContinuation = eventContinuation
        self.frames = frames
        self.frameContinuation = frameContinuation
    }

    /// Start the generation loop. Must be called once after init.
    func start() {
        generationTask = Task { [weak self] in
            guard let self else { return }
            await self.generateLoop()
        }
        frameContinuation.onTermination = { [weak self] _ in
            guard let self else { return }
            Task { await self.cancel() }
        }
        eventContinuation.onTermination = { [weak self] _ in
            guard let self else { return }
            Task { await self.cancel() }
        }
    }

    // MARK: - Generation Loop

    private func generateLoop() async {
        var utteranceIndex = 0

        do {
            for await text in textStream {
                if Task.isCancelled { break }

                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }

                let plan = PocketTtsSynthesizer.makeTextPlan(
                    trimmed, tokenizer: constants.tokenizer,
                    maxTokens: maxTokensPerChunk, language: language
                )
                let chunks = plan.chunks
                Self.logger.info(
                    "Session enqueued '\(trimmed)', \(chunks.count) chunk(s)")
                eventContinuation.yield(.utterancePlanned(utteranceIndex: utteranceIndex, plan: plan))

                for chunk in chunks {
                    if Task.isCancelled { break }

                    try await generateChunk(
                        chunk: chunk,
                        chunkCount: chunks.count,
                        utteranceIndex: utteranceIndex
                    )
                }
                utteranceIndex += 1
            }
            eventContinuation.finish()
            frameContinuation.finish()
        } catch {
            eventContinuation.finish(throwing: error)
            frameContinuation.finish(throwing: error)
        }
    }

    private func generateChunk(
        chunk: PocketTtsSynthesizer.PlanChunk,
        chunkCount: Int,
        utteranceIndex: Int
    ) async throws {
        let text = chunk.synthesisText
        let isMidSentence = chunk.isMidSentence
        let chunkIndex = chunk.id
        if stateModels != nil {
            // `.aneState`: MLState multifunction pipeline. The store refuses
            // to load these models on pre-15/18 OSes, so this gate is
            // unreachable in practice — it exists for the compiler.
            guard #available(macOS 15.0, iOS 18.0, *) else {
                throw PocketTTSError.processingFailed(
                    "PocketTTS `.aneState` placement requires macOS 15+/iOS 18+")
            }
            try await generateChunkStateful(
                chunk: chunk, chunkCount: chunkCount,
                utteranceIndex: utteranceIndex)
            return
        }

        guard let condModel, let stepModel, let flowModel,
            let condLayerKeys, let flowlmLayerKeys, let voiceKVSnapshot
        else {
            throw PocketTTSError.processingFailed(
                "PocketTTS session misconfigured: IO models missing")
        }

        let (normalizedChunk, framesAfterEos) = PocketTtsSynthesizer.normalizeText(
            text, isMidSentence: isMidSentence, language: language)
        Self.logger.info("Session chunk \(chunkIndex): '\(normalizedChunk)'")

        // Tokenize and embed
        let tokenIds = constants.tokenizer.encode(normalizedChunk)
        let textEmbeddings = PocketTtsSynthesizer.embedTokens(tokenIds, constants: constants)

        // Clone voice KV snapshot and prefill text tokens only
        var kvState = try PocketTtsSynthesizer.cloneKVCacheState(voiceKVSnapshot)
        kvState = try await PocketTtsSynthesizer.prefillKVCacheText(
            state: kvState, textEmbeddings: textEmbeddings, model: condModel,
            layerKeys: condLayerKeys,
            prefillModel: condPrefillModel ?? condModel, prefillLayerKeys: condPrefillLayerKeys,
            useFastPrefill: useCondPrefill
        )

        // Generation loop
        let maxGenLen = PocketTtsSynthesizer.estimateMaxFrames(text: text)
        var eosStep: Int?
        var sequence = try PocketTtsSynthesizer.createBosStartSequence(
            bosEmbedding: constants.bosEmbedding,
            splitKV: flowlmLayerKeys.isSplitKV)
        let totalFramesAfterEos = framesAfterEos + PocketTtsConstants.extraFramesAfterDetection
        var generatedSampleCount = 0

        for step in 0..<maxGenLen {
            if Task.isCancelled { break }

            // FlowLM step. `kvState` is function-local (not actor-isolated),
            // so it can be passed `inout` to the async free function directly.
            let (transformerOut, eosLogit) = try await PocketTtsSynthesizer.runFlowLMStep(
                sequence: sequence,
                bosEmb: bosEmb,
                state: &kvState,
                model: stepModel,
                layerKeys: flowlmLayerKeys
            )

            // EOS detection
            if eosLogit > PocketTtsConstants.eosThreshold && eosStep == nil {
                eosStep = step
                Self.logger.info("Session chunk \(chunkIndex) EOS at step \(step)")
            }
            if let eos = eosStep, step >= eos + totalFramesAfterEos {
                break
            }

            // Flow decode with actor-isolated RNG
            var localRng = rng
            let latent = try await PocketTtsSynthesizer.flowDecode(
                transformerOut: transformerOut,
                temperature: temperature,
                model: flowModel,
                rng: &localRng
            )
            rng = localRng

            // Mimi decode with actor-isolated state
            var localMimi = mimiState
            let frameSamples = try await PocketTtsSynthesizer.runMimiDecoder(
                latent: latent,
                state: &localMimi,
                model: mimiModel,
                mimiKeys: mimiKeys
            )
            mimiState = localMimi

            // Yield frame via the events stream. Oratio consumes `events`;
            // duplicating sample arrays into the legacy `frames` stream would
            // retain an unbounded unused buffer for events-only consumers.
            let frame = PocketTtsSynthesizer.AudioFrame(
                samples: frameSamples,
                frameIndex: step,
                chunkIndex: chunkIndex,
                chunkCount: chunkCount,
                utteranceIndex: utteranceIndex
            )
            generatedSampleCount += frameSamples.count
            eventContinuation.yield(.audioFrame(frame))

            // Autoregressive feedback
            sequence = try PocketTtsSynthesizer.createSequenceFromLatent(latent)
        }

        emitHighlights(for: chunk, generatedSampleCount: generatedSampleCount, utteranceIndex: utteranceIndex)
    }

    /// Emit estimated word/reading highlight spans for a completed chunk.
    private func emitHighlights(
        for chunk: PocketTtsSynthesizer.PlanChunk,
        generatedSampleCount: Int,
        utteranceIndex: Int
    ) {
        let audioDuration = TimeInterval(generatedSampleCount) / TimeInterval(PocketTtsConstants.audioSampleRate)
        let spans = PocketTtsSynthesizer.estimatedHighlightSpans(for: chunk, audioDuration: audioDuration)
        guard spans.isEmpty == false else { return }
        eventContinuation.yield(
            .chunkHighlights(
                utteranceIndex: utteranceIndex,
                chunkIndex: chunk.id,
                audioDuration: audioDuration,
                spans: spans
            ))
    }

    // MARK: - Stateful (`.aneState`) Generation — mobius Trial 23

    /// The lazily-created `PocketTtsStateEngine` for this session. One engine
    /// (and one underlying `MLState`) per session, reused across utterances.
    @available(macOS 15.0, iOS 18.0, *)
    private func stateEngine() throws -> PocketTtsStateEngine {
        if let engine = stateEngineRef as? PocketTtsStateEngine {
            return engine
        }
        guard let stateModels else {
            throw PocketTTSError.processingFailed(
                "PocketTTS state session missing multifunction models")
        }
        let engine = PocketTtsStateEngine(
            models: stateModels, layers: language.transformerLayers)
        stateEngineRef = engine
        return engine
    }

    /// The session voice's KV snapshot in the state's fp16 at-rest format.
    /// Computed once: shipped voices convert their pre-baked snapshot
    /// directly; cloned voices run the `[bos_before_voice, voice...]` block
    /// through the stateful prefill once and capture the resulting state.
    @available(macOS 15.0, iOS 18.0, *)
    private func stateVoiceFp16Snapshot(
        engine: PocketTtsStateEngine
    ) async throws -> PocketTtsStateEngine.Fp16Snapshot {
        if let cached = stateVoiceSnapshotRef as? PocketTtsStateEngine.Fp16Snapshot {
            return cached
        }
        guard let voiceData = stateVoiceData else {
            throw PocketTTSError.processingFailed(
                "PocketTTS state session missing voice data")
        }

        let snapshot: PocketTtsStateEngine.Fp16Snapshot
        if let baked = voiceData.cacheSnapshot {
            snapshot = try PocketTtsStateEngine.fp16Snapshot(
                from: baked, layers: language.transformerLayers)
        } else if voiceData.promptLength > 0 {
            guard let bosBeforeVoice = constants.bosBeforeVoice else {
                throw PocketTTSError.processingFailed(
                    "PocketTTS cloned-voice prefill requires bos_before_voice constant. "
                        + "Re-download the language pack (FluidAudio #592 fix).")
            }
            let dim = PocketTtsConstants.embeddingDim
            var flat = [Float]()
            flat.reserveCapacity((1 + voiceData.promptLength) * dim)
            flat.append(contentsOf: bosBeforeVoice)
            flat.append(contentsOf: voiceData.audioPrompt[0..<(voiceData.promptLength * dim)])
            try await engine.resetToZero()
            try await engine.prefill(
                flatConditioning: flat, tokenCount: 1 + voiceData.promptLength)
            snapshot = try await engine.captureSnapshot()
        } else {
            // Empty voice prompt: chunks start from a zeroed cache.
            try await engine.resetToZero()
            snapshot = try await engine.captureSnapshot()
        }
        stateVoiceSnapshotRef = snapshot
        Self.logger.info(
            "Session state voice snapshot ready at position \(Int(snapshot.position))")
        return snapshot
    }

    /// Stateful counterpart of `generateChunk`: voice injection (= cache
    /// reset) + ONE prefill call + one fused `generate` dispatch per frame.
    /// Mimi decoding, EOS handling, and frame emission are identical to the
    /// IO path. The flow-decoder noise comes from the same seeded RNG so
    /// `--seed` reproducibility holds (note: the fused step consumes its 32
    /// noise draws BEFORE the post-EOS break is evaluated, so the draw
    /// stream differs from the IO placements after the final frame — seeds
    /// are reproducible within a placement, not across placements).
    @available(macOS 15.0, iOS 18.0, *)
    private func generateChunkStateful(
        chunk: PocketTtsSynthesizer.PlanChunk,
        chunkCount: Int,
        utteranceIndex: Int
    ) async throws {
        let text = chunk.synthesisText
        let isMidSentence = chunk.isMidSentence
        let chunkIndex = chunk.id
        let engine = try stateEngine()
        let voiceSnapshot = try await stateVoiceFp16Snapshot(engine: engine)

        let (normalizedChunk, framesAfterEos) = PocketTtsSynthesizer.normalizeText(
            text, isMidSentence: isMidSentence, language: language)
        Self.logger.info("Session chunk \(chunkIndex) (state): '\(normalizedChunk)'")

        // Tokenize and embed
        let tokenIds = constants.tokenizer.encode(normalizedChunk)
        let textEmbeddings = PocketTtsSynthesizer.embedTokens(tokenIds, constants: constants)
        let dim = PocketTtsConstants.embeddingDim
        var flatText = [Float]()
        flatText.reserveCapacity(textEmbeddings.count * dim)
        for embedding in textEmbeddings { flatText.append(contentsOf: embedding) }

        // Voice injection doubles as the per-chunk cache reset (it
        // overwrites all 512 slots), then one prefill call for the text.
        try await engine.reset(with: voiceSnapshot)
        try await engine.prefill(
            flatConditioning: flatText, tokenCount: textEmbeddings.count)
        let prefilledPosition = await engine.position
        Self.logger.info("State KV prefilled to position \(Int(prefilledPosition))")

        // Generation loop. BOS = the BOS latent passed as `sequence`
        // (the fused graph has no NaN-BOS protocol — same contract as the
        // rank-4 `.ane` models).
        let maxGenLen = PocketTtsSynthesizer.estimateMaxFrames(text: text)
        var eosStep: Int?
        var sequence = constants.bosEmbedding
        let totalFramesAfterEos = framesAfterEos + PocketTtsConstants.extraFramesAfterDetection
        var generatedSampleCount = 0

        for step in 0..<maxGenLen {
            if Task.isCancelled { break }

            // z_0 noise from the session RNG (same draw count and scaling
            // as the IO path's flowDecode).
            var localRng = rng
            let scale = sqrtf(temperature)
            var noise = [Float](repeating: 0, count: PocketTtsConstants.latentDim)
            for i in 0..<noise.count {
                noise[i] = Float.gaussianRandom(using: &localRng) * scale
            }
            rng = localRng

            let (latent, eosLogit) = try await engine.generateFrame(
                sequence: sequence, noise: noise)

            // EOS detection
            if eosLogit > PocketTtsConstants.eosThreshold && eosStep == nil {
                eosStep = step
                Self.logger.info("Session chunk \(chunkIndex) (state) EOS at step \(step)")
            }
            if let eos = eosStep, step >= eos + totalFramesAfterEos {
                break
            }

            // Mimi decode with actor-isolated state
            var localMimi = mimiState
            let frameSamples = try await PocketTtsSynthesizer.runMimiDecoder(
                latent: latent,
                state: &localMimi,
                model: mimiModel,
                mimiKeys: mimiKeys
            )
            mimiState = localMimi

            // Yield frame via the events stream (see IO path rationale).
            let frame = PocketTtsSynthesizer.AudioFrame(
                samples: frameSamples,
                frameIndex: step,
                chunkIndex: chunkIndex,
                chunkCount: chunkCount,
                utteranceIndex: utteranceIndex
            )
            generatedSampleCount += frameSamples.count
            eventContinuation.yield(.audioFrame(frame))

            // Autoregressive feedback
            sequence = latent
        }

        emitHighlights(for: chunk, generatedSampleCount: generatedSampleCount, utteranceIndex: utteranceIndex)
    }
}
