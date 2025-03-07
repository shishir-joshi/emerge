def find_chunk_spans(transcript: str, offset_mapping: list, chunk: str) -> list:
    """
    Find token span indices of chunk within transcript.
    
    Args:
        transcript: Full conversation transcript
        chunk: Text segment to locate
    
    Returns:
        List of (start, end) token indices
    """
    # offset_mapping = offset_mapping[0]
    
    # Find chunk in transcript
    chunk_start = transcript.find(chunk)
    if chunk_start == -1:
        return []
    chunk_end = chunk_start + len(chunk)
    
    # Find token spans that overlap with chunk
    span_start = None
    span_end = None
    
    for idx, (start, end) in enumerate(offset_mapping):
        if start == end:  # Skip special tokens
            continue
            
        if span_start is None and end > chunk_start:
            span_start = idx
        if span_start is not None and start >= chunk_end:
            span_end = idx
            break
            
    if span_start is not None and span_end is None:
        span_end = len(offset_mapping)
        
    return [(span_start, span_end)] if span_start is not None else []