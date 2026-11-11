import sys
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import doc_chunked

def embedding (text_chunked : list[str], model):
   
    embeddings =model.encode(text_chunked, normalize_embeddings=True)

    return embeddings

if __name__ == "__main__":
    
    if len(sys.argv) == 2 :
        doc = sys.argv[1]
    else :
        print(f"Usage : {sys.argv[0]} doc.pdf")

    chunks = doc_chunked(doc)

    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    embed = embedding(chunks, model)

    q = "Quel est la distance total sur les burpee broad jump"
    q_em = model.encode(q, normalize_embeddings=True)

    scores = embed @ q_em 

    for i in np.argsort(scores)[::-1][:3] : 
        print(f"{scores[i]} - Chunk {i} : {chunks[i]}")