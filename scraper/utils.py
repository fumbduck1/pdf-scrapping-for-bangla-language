import re


def _sentence_chunks(text: str):
    """
    Split text into rough sentence or clause chunks using Bangla and English sentence punctuation.
    
    If `text` is falsy, returns an empty list. Splits on Bangla danda (।), period (.), question mark (?) and exclamation mark (!), and returns non-empty, trimmed chunks.
    
    Parameters:
        text (str): Input text to split.
    
    Returns:
        list[str]: Non-empty, trimmed sentence or clause chunks.
    """
    if not text:
        return []
        
    punc_pattern = re.compile(r'(?<=[।.?!])\s*')
    chunks = punc_pattern.split(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]