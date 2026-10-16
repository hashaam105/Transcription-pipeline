"""Render a transcription result as SRT, VTT or timestamped text."""
from .transcriber import format_ts


def to_srt(result: dict) -> str:
    return "\n".join(
        f"{i}\n{format_ts(s['start'], ',')} --> {format_ts(s['end'], ',')}\n{s['text']}\n"
        for i, s in enumerate(result["segments"], start=1)
    )


def to_vtt(result: dict) -> str:
    body = "\n".join(
        f"{format_ts(s['start'])} --> {format_ts(s['end'])}\n{s['text']}\n" for s in result["segments"]
    )
    return f"WEBVTT\n\n{body}"


def to_txt(result: dict) -> str:
    return "\n".join(f"[{s['start_ts']} - {s['end_ts']}] {s['text']}" for s in result["segments"])


FORMATTERS = {
    "srt": (to_srt, "application/x-subrip"),
    "vtt": (to_vtt, "text/vtt"),
    "txt": (to_txt, "text/plain"),
}
