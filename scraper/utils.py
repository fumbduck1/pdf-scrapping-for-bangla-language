import re


def _sentence_chunks(text: str):
    """Split text into rough sentences/clauses using Bangla/English punctuation."""
    if not text:
        return []
    # Normalize whitespace
    cleaned = '\n'.join(' '.join(line.split()) for line in text.splitlines())
    # Split on Bangla danda or common sentence enders
    parts = re.split(r"(?<=[।!?])\s+", cleaned)
    return [p.strip() for p in parts if p and p.strip()]