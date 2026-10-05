# Oratio fork changes

This branch (`oratio-fork-v2.1` of `sergdort/FluidAudio`) is a modified version of
[FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio), based on upstream
commit `1a2da181d1a33ca125ec1bd0f25790f41eb5fa54`. It is used by the Oratio app and is
distributed under the same [Apache License 2.0](LICENSE). The changes are by Serg Dort (2026)
and are limited to PocketTTS.

## Modified files

Each modified file starts with a comment describing the change.

- `Sources/FluidAudio/TTS/PocketTTS/Pipeline/PocketTtsSession.swift` — `events` stream with
  text plans, audio frames, and estimated highlights; audio is delivered through it.
- `Sources/FluidAudio/TTS/PocketTTS/Pipeline/PocketTtsSynthesizer.swift` — speakable
  abbreviation expansion (e.g./i.e.), trailing-quote-aware terminal punctuation, and
  abbreviation-aware sentence splitting.
- `Sources/FluidAudio/TTS/PocketTTS/Tokenizer/SentencePieceProto.swift` — piece types and the
  byte-fallback setting.
- `Sources/FluidAudio/TTS/PocketTTS/Tokenizer/SentencePieceTokenizer.swift` — byte-fallback
  tokenization.
- `Tests/FluidAudioTests/TTS/PocketTTS/SentencePieceProtoTests.swift` — tests for the above.

## Added files

- `Sources/FluidAudio/TTS/PocketTTS/Pipeline/PocketTtsSynthesizer+TextPlan.swift` — text plans
  that map synthesized audio chunks to source-text ranges, and estimated word highlights.
- `Tests/FluidAudioTests/TTS/PocketTTS/PocketTtsHighlightTests.swift`
- `Tests/FluidAudioTests/TTS/PocketTTS/PocketTtsTextPlanTests.swift`

The commit history of this branch records every change in detail.
