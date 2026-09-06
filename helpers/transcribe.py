"""Transcribe a video to word-level JSON.

Extracts mono 16kHz audio via ffmpeg, runs it through one of three engines,
and writes the result to <edit_dir>/transcripts/<video_stem>.json.

Engines (see engines.py for the shared payload shape):

    mlx-whisper  default. Local, free, word-level. Apple Silicon GPU.
                 No diarization: every word is speaker_0.
    vibevoice    local, free, diarized, supports hotwords, 50+ languages.
                 Timestamps are per utterance, NOT per word, so cuts from it
                 land on utterance boundaries. Needs a git-clone install.
    elevenlabs   hosted, paid. Word-level, diarized, tags audio events
                 ((laughs), (sighs)) that the other two do not produce.

Cached: if the output file already exists, the transcription is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --engine elevenlabs
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --num-speakers 2
    python helpers/transcribe.py <video_path> --hotwords "Xugan Chen,video-use"
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import engines


def load_api_key() -> str:
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "ELEVENLABS_API_KEY":
                    return v.strip().strip('"').strip("'")
    v = os.environ.get("ELEVENLABS_API_KEY", "")
    if not v:
        sys.exit("ELEVENLABS_API_KEY not found in .env or environment")
    return v


def count_audio_tracks(video_path: Path) -> int:
    """How many audio streams the container holds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def peak_dbfs(wav_path: Path) -> float:
    """Peak level of a 16-bit PCM wav, in dBFS. -inf for digital silence."""
    peak = 0
    with wave.open(str(wav_path), "rb") as w:
        # A chunk at a time: batch mode runs several of these at once, and a two-hour
        # take is 230 MB of 16 kHz mono before the array copy doubles it.
        while frames := w.readframes(1 << 16):
            samples = array.array("h", frames)
            peak = max(peak, max(samples), -min(samples))
    return 20 * math.log10(peak / 32768) if peak > 0 else float("-inf")


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:a:{audio_track}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcript_path(
    edit_dir: Path,
    video: Path,
    audio_track: int = 0,
    engine: str | None = None,
) -> Path:
    """Where a video's transcript lands.

    The track belongs in the name, or a rerun with --audio-track hands back the transcript of
    the track it is meant to replace. Track 0 keeps the plain name, so transcripts made before
    the flag existed stay valid. Batch mode tests its cache with this too — one function, so
    the two cannot drift apart.

    The engine belongs in the name for the same reason: two engines produce different word
    lists for the same audio, and the cache must not hand back the wrong one. `elevenlabs`
    keeps the plain name so transcripts made before --engine existed stay valid.
    """
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    if engine and engine != "elevenlabs":
        suffix += f".{engine}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str | None = None,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    audio_track: int = 0,
    engine: str = engines.DEFAULT_ENGINE,
    hotwords: list[str] | None = None,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_path(edit_dir, video, audio_track, engine)

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    n_tracks = count_audio_tracks(video)
    if n_tracks > 1 and verbose:
        print(f"  note: {video.name} has {n_tracks} audio tracks, using track "
              f"{audio_track + 1} (--audio-track to change)", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio, audio_track)

        # Transcribing silence returns nothing and, on the hosted engine, costs the
        # same as speech. Catch the wrong-track case before spending anything.
        peak = peak_dbfs(audio)
        if peak < -60.0:
            raise RuntimeError(
                f"track {audio_track + 1} of {video.name} is silent "
                f"(peak {peak:.1f} dBFS) - not transcribing. "
                + (f"The file has {n_tracks} audio tracks; try --audio-track "
                   + " or ".join(str(i) for i in range(n_tracks) if i != audio_track) + "."
                   if n_tracks > 1 else "Check the source audio.")
            )

        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  {engine}: {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
        payload = engines.transcribe_audio(
            engine,
            audio,
            language=language,
            num_speakers=num_speakers,
            api_key=api_key,
            hotwords=hotwords,
        )

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")
        if payload.get("timestamp_granularity") == "segment":
            print("    note: utterance-level timestamps — cuts will snap to "
                  "utterance edges, not words.")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video to word-level JSON")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--engine",
        choices=engines.ENGINES,
        default=engines.DEFAULT_ENGINE,
        help=f"Transcription engine (default: {engines.DEFAULT_ENGINE}). "
             "mlx-whisper is local, free and word-level; vibevoice is local and "
             "diarized but utterance-level; elevenlabs is hosted, paid, word-level "
             "and the only one that tags audio events.",
    )
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy. "
             "Only used by engines that diarize (elevenlabs, vibevoice).",
    )
    ap.add_argument(
        "--hotwords",
        type=str,
        default=None,
        help="Comma-separated names or domain terms to bias recognition toward. "
             "Supported by vibevoice and elevenlabs (billed +20%% on elevenlabs).",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe. OBS writes the game on track 0 "
             "and the mic on track 1; without this ffmpeg applies its default audio "
             "stream selection, which picks the track with the most channels.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    api_key = load_api_key() if engines.needs_api_key(args.engine) else None
    hotwords = [h.strip() for h in args.hotwords.split(",") if h.strip()] if args.hotwords else None

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=api_key,
        language=args.language,
        num_speakers=args.num_speakers,
        audio_track=args.audio_track,
        engine=args.engine,
        hotwords=hotwords,
    )


if __name__ == "__main__":
    main()
