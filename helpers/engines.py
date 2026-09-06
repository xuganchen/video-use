"""Speech-to-text engines.

Every engine returns the same payload shape, the one the rest of the pipeline
already reads (pack_transcripts, timeline_view, render):

    {
      "language_code": "zho",
      "text": "...",
      "words": [{"text", "start", "end", "type", "speaker_id"}, ...],
      "audio_duration_secs": 2922.4,
      "engine": "mlx-whisper",
      "timestamp_granularity": "word" | "segment",
    }

`type` is "word", "spacing", or "audio_event". render.py keeps only "word";
pack_transcripts also renders "audio_event" and breaks phrases on gaps.

`timestamp_granularity` is the honest flag. Cuts snap to word boundaries
(SKILL.md Hard Rules 6 and 8), so a "segment" transcript can only cut at
utterance edges. Nothing fabricates word times it does not have.
"""

from __future__ import annotations

import wave
from pathlib import Path

DEFAULT_ENGINE = "mlx-whisper"
ENGINES = ("mlx-whisper", "vibevoice", "elevenlabs")

# Whisper large-v3, MLX-converted. Runs on the Apple Silicon GPU.
MLX_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


# --------------------------------------------------------------------------
# mlx-whisper (default): local, free, word-level, no diarization
# --------------------------------------------------------------------------


def _mlx_whisper(
    audio: Path,
    language: str | None = None,
    model: str = MLX_WHISPER_MODEL,
    **_: object,
) -> dict:
    try:
        import mlx_whisper
    except ImportError as e:
        raise RuntimeError(
            "mlx-whisper is not installed. Run: pip install mlx-whisper\n"
            "(Apple Silicon only. On other hardware use --engine elevenlabs.)"
        ) from e

    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=model,
        word_timestamps=True,
        language=language,
        verbose=None,
    )

    words: list[dict] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            if not text:
                continue
            words.append({
                "text": text,
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
                "type": "word",
                # Whisper does no diarization. One speaker keeps the schema honest
                # rather than inventing turns; pack_transcripts only breaks on
                # speaker change when the ids actually differ.
                "speaker_id": "speaker_0",
            })

    return {
        "language_code": result.get("language"),
        "text": (result.get("text") or "").strip(),
        "words": words,
        "audio_duration_secs": wav_duration(audio),
        "engine": "mlx-whisper",
        "timestamp_granularity": "word",
    }


# --------------------------------------------------------------------------
# VibeVoice-ASR: local, free, diarized, hotwords — but segment-level only
# --------------------------------------------------------------------------


def _vibevoice(
    audio: Path,
    language: str | None = None,
    hotwords: list[str] | None = None,
    **_: object,
) -> dict:
    """VibeVoice-ASR-7B. Long-form, diarized, supports hotwords.

    Timestamps are per utterance, not per word, so each segment becomes one
    entry in `words`. Cuts made from this transcript land on utterance
    boundaries. Use it for multi-speaker material where knowing who spoke
    matters more than frame-accurate cut points; use mlx-whisper otherwise.
    """
    install_hint = (
        "VibeVoice is not installed.\n"
        "  git clone https://github.com/microsoft/VibeVoice.git\n"
        "  cd VibeVoice && pip install -e .\n"
        "The `vibevoice` on PyPI is NOT Microsoft's — it must come from the repo.\n"
        "The 7B model is a ~18GB download and the project targets CUDA; on Apple\n"
        "Silicon it runs through MPS without flash-attn and is slow.\n"
        "For word-level timestamps use --engine mlx-whisper instead."
    )
    # The vibevoice package is what registers the custom architecture with
    # transformers. Without it, from_pretrained fails deep inside transformers
    # with an unrecognized-model-type ValueError, after a pointless network
    # round-trip — so check for the package itself, not just its dependencies.
    try:
        import vibevoice  # noqa: F401
    except ImportError as e:
        raise RuntimeError(install_hint) from e

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as e:
        raise RuntimeError("VibeVoice needs torch and transformers installed.") from e

    model_id = "microsoft/VibeVoice-ASR"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except ValueError as e:
        raise RuntimeError(f"{install_hint}\n\nunderlying error: {e}") from e

    inputs = processor(
        audio=str(audio),
        hotwords=hotwords or None,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=8192)

    generated = processor.decode(
        output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    segments = processor.post_process_transcription(generated)

    words: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        start, end = seg.get("start_time"), seg.get("end_time")
        if not text or start is None or end is None:
            continue
        words.append({
            "text": text,
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "type": "word",
            "speaker_id": str(seg.get("speaker_id", "speaker_0")),
        })

    return {
        "language_code": language,
        "text": " ".join(w["text"] for w in words),
        "words": words,
        "audio_duration_secs": wav_duration(audio),
        "engine": "vibevoice",
        "timestamp_granularity": "segment",
    }


# --------------------------------------------------------------------------
# ElevenLabs Scribe: hosted, paid, word-level, diarized, audio events
# --------------------------------------------------------------------------

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def _elevenlabs(
    audio: Path,
    language: str | None = None,
    num_speakers: int | None = None,
    api_key: str | None = None,
    hotwords: list[str] | None = None,
    model_id: str = "scribe_v1",
    **_: object,
) -> dict:
    import json as _json

    import requests

    if not api_key:
        raise RuntimeError("ElevenLabs engine needs an API key; run `video-use key`.")

    data: dict[str, str] = {
        "model_id": model_id,
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)
    if hotwords:
        # Biases recognition toward names and domain terms. Billed at +20%.
        data["keyterms"] = _json.dumps(hotwords)

    with open(audio, "rb") as f:
        resp = requests.post(
            SCRIBE_URL,
            headers={"xi-api-key": api_key},
            files={"file": (audio.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Scribe returned {resp.status_code}: {resp.text[:500]}")

    payload = resp.json()
    payload.setdefault("audio_duration_secs", wav_duration(audio))
    payload["engine"] = "elevenlabs"
    payload["timestamp_granularity"] = "word"
    return payload


_DISPATCH = {
    "mlx-whisper": _mlx_whisper,
    "vibevoice": _vibevoice,
    "elevenlabs": _elevenlabs,
}


def needs_api_key(engine: str) -> bool:
    return engine == "elevenlabs"


def transcribe_audio(engine: str, audio: Path, **kwargs: object) -> dict:
    """Run one engine over a wav file and return the common payload."""
    fn = _DISPATCH.get(engine)
    if fn is None:
        raise RuntimeError(f"unknown engine: {engine} (choose from {', '.join(ENGINES)})")
    return fn(audio, **kwargs)  # type: ignore[arg-type]
