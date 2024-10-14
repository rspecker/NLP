import numpy as np
from sentence_transformers import SentenceTransformer


def create_sentence_embeddings(sentences: list[str], model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generates embeddings for a list of sentences using a pre-trained Sentence 
    Transformer model.

    Args:
        sentences: A list of sentences for which to calculate embeddings.
        model: The name of the Sentence Transformer model to use. The model
            must be supported by the Sentence Transformer library.

    Returns:
        A 2D array where each row corresponds to the embedding of the 
        respective sentence.
    """
    # Load a pretrained Sentence Transformer model
    model = SentenceTransformer(model)
    
    # Calculate embeddings by calling model.encode()
    embeddings = model.encode(sentences)
    
    return embeddings
