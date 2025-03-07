import torch

def late_chunking(model_output: 'BatchEncoding', span_annotation: list, max_length=None):
    """
    Perform late chunking on model output based on span annotations.

    Args:
        model_output (BatchEncoding): The output from the model containing token embeddings.
        span_annotation (list): List of span annotations indicating the start and end of each chunk.
        max_length (int, optional): Maximum length of the model. Annotations beyond this length will be ignored.

    Returns:
        list: List of pooled embeddings for each chunk.
    """
    # Extract token embeddings from model output
    # We need these embeddings to perform chunking based on the provided span annotations
    token_embeddings = model_output[0]
    outputs = []

    # Iterate over each set of embeddings and corresponding annotations
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if max_length is not None:
            # Adjust annotations to ensure they do not exceed the maximum length of the model
            # This is important to avoid indexing errors and to respect the model's maximum input length
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        
        # Pool embeddings for each chunk by averaging the embeddings within the span
        # Averaging the embeddings helps in creating a single representation for each chunk
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1  # Ensure the span length is at least 1 to avoid division by zero
        ]
        
        # Detach embeddings from the computation graph and convert to numpy arrays
        # Detaching is necessary to prevent memory leaks and to allow for further processing outside of PyTorch
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        
        # Append the pooled embeddings to the output list
        # This list will contain the final chunked embeddings for all input sequences
        outputs.append(pooled_embeddings)

    return outputs