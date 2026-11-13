import sys
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from chunking import doc_chunked


TOP_K = 10
BI_ENCODER_NAME = "paraphrase-multilingual-mpnet-base-v2"
RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

def embed_texts (texts : list[str], model):
    return model.encode(texts, normalize_embeddings=True)

def embed_chunks (chunks, model) :
    texts = [c['text'] for c in chunks]
    return embed_texts(texts, model)

def rerank_scores(chunks, question, top_index, reranker) :
    pairs = [(question, chunks[i]['text']) for i in top_index]
    return reranker.predict(pairs)

def create_models():
    return SentenceTransformer(BI_ENCODER_NAME), CrossEncoder(RERANKER_NAME)

if __name__ == "__main__":
    
    question = input("Écrit la question que tu souhaites poser :")
    while not question.strip() :
        question = input("Écrit la question que tu souhaites poser :")

    path_pdf = "../data/26_27_HYROX_RulebookDoubles_FR_210726.pdf"

    chunks = doc_chunked(path_pdf)

    bi_encoder, reranker = create_models()

    chunk_embeddings = embed_chunks(chunks, bi_encoder)

    question_embedding = bi_encoder.encode(question, normalize_embeddings=True)

    bi_scores = chunk_embeddings @ question_embedding

    top_index = np.argsort(bi_scores)[::-1][:TOP_K]

    new_scores = rerank_scores(chunks, question, top_index, reranker)

    for i in np.argsort(new_scores)[::-1][:3] :
        print(f"{new_scores[i]} Chunk {top_index[i]}: {chunks[top_index[i]]}")
        print()