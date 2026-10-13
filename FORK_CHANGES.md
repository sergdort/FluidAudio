# Oratio fork changes

This branch (`oratio-fork-v2.2` of `sergdort/FluidAudio`) is a modified version of
[FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio), based on upstream
commit `793a2c5a0e99705a55fefbad00e65ca684329349`. It is used by the Oratio app and is
distributed under the same [Apache License 2.0](LICENSE). The changes are by Serg Dort (2026)
and are limited to PocketTTS.

## Modified files

Each modified file starts with a comment describing the change.

- `Sources/FluidAudio/TTS/PocketTTS/Pipeline/PocketTtsSession.swift` — `events` stream with
  text plans, audio frames, and estimated highlights; audio is delivered through it. Text is
  planned with a voice-independent cache budget.
- `Sources/FluidAudio/TTS/PocketTTS/Pipeline/PocketTtsSynthesizer.swift` — speakable
  abbreviation expansion (e.g./i.e.), trailing-quote-aware terminal punctuation, and
  abbreviation-aware sentence splitting.
- `Sources/FluidAudio/TTS/PocketTTS/PocketTtsConstants.swift` — voice-independent planning
  cache budget.
- `Tests/FluidAudioTests/TTS/PocketTTS/SentencePieceTokenizerTests.swift` — byte-fallback
  lattice test.

## Added files

- `Sources/FluidAudio/TTS/PocketTTS/Pipeline/PocketTtsSynthesizer+TextPlan.swift` — text plans
  that map synthesized audio chunks to source-text ranges (keeping cache-safe sentences
  whole), and estimated word highlights.
- `Tests/FluidAudioTests/TTS/PocketTTS/PocketTtsHighlightTests.swift`
- `Tests/FluidAudioTests/TTS/PocketTTS/PocketTtsTextPlanTests.swift`

The commit history of this branch records every change in detail.
