import re


def _sentence_chunks(text: str):
    """Split text into rough sentences/clauses using Bangla/English punctuation."""
    if not text:
        return []
        
    punc_pattern = re.compile(r'(?<=[।.?!])\s*')
    chunks = punc_pattern.split(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]
