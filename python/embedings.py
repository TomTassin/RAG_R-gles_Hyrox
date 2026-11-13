import sys
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import doc_chunked

def embedding (text_chunked : list[str], model):
   
    embeddings = model.encode(text_chunked, normalize_embeddings=True)

    return embeddings

def embed_chunks (chunks, model) :
    c_text = [c['text'] for c in chunks]

    return embedding(c_text, model)

if __name__ == "__main__":
    
    if len(sys.argv) == 2 :
        doc = sys.argv[1]
    else :
        print(f"Usage : {sys.argv[0]} doc.pdf")
        sys.exit(1)

    chunks = doc_chunked(doc)

    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


    embed_text = embed_chunks(chunks, model)

    q = "Quel est la distance total sur les burpee broad jump"
    q_em = model.encode(q, normalize_embeddings=True)

    scores_text = embed_text @ q_em

    for i in np.argsort(scores_text)[::-1][:3] : 
            print(f"{scores_text[i]} - Chunk {i} : {chunks[i]}")
    