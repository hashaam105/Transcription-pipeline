"""Text clean-up for downstream use."""
import re

FILLER_PATTERN = re.compile(r"\b(um+|uh+|erm|hmm+|you know)\b[,]?\s*", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = FILLER_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    if text:
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."
    return text


def word_count(text: str) -> int:
    return len(text.split())
