"""Command-line entry point.

    python -m transcription_service.cli sample.mp3 --format srt --output sample.srt
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from .audio import AudioValidationError
from .formatters import FORMATTERS
from .pipeline import TranscriptionPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file to text.")
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--format", choices=["json", "srt", "vtt", "txt"], default="json")
    parser.add_argument("--language", default=None, help="ISO code, e.g. 'en'. Default: auto-detect")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--output", type=Path, help="Write output to a file instead of stdout")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    try:
        result = TranscriptionPipeline().run(args.audio_file, language=args.language,
                                             word_timestamps=args.word_timestamps)
    except AudioValidationError as e:
        logging.error("Invalid input: %s", e)
        return 2

    if args.format == "json":
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output = FORMATTERS[args.format][0](result["transcription"])

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        logging.info("Saved %s", args.output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
