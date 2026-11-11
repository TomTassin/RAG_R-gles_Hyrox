from pypdf import PdfReader
import numpy as np
import re
import sys


NOISE = {
    re.compile(r'HYROX\s.*_26/27([ \t]+\d+)?[ \t]*'),
    re.compile(r'^[ \t]*\d{1,2}[ \t]*(cm)?[ \t]*$', re.MULTILINE),
    re.compile(r'FIG\s*\d+'),
}

def clean_text(text : str) -> str :
    text_cleaned = text
    for r in NOISE :
        text_cleaned = re.sub(r, "", text_cleaned)
    text_cleaned = re.sub(r'[ \t]+', ' ', text_cleaned)
    text_cleaned = re.sub(r'\n{3,}', '\n\n', text_cleaned)
    return text_cleaned

def chunk_text (text : str, chunk_size : int = 1) -> list[str]:
    """ 
    chunk_size  : number of sentence per chunk
    """
    text = clean_text(text)

    sentences = re.split(r'(?<=[?.!])\s+|\n\s*\n', text)
    chunks = []
    start = 0

    while start < len(sentences) : 
        end = min(start + chunk_size, len(sentences))
        chunk = ' '.join(sentences[start : end]).strip()
        if chunk :
            chunks.append(chunk)
        start += chunk_size

    return chunks

def doc_chunked (doc : str) -> list[str]:

    reader = PdfReader(doc)

    text = ""
    for i,page in enumerate(reader.pages) :
        page = page.extract_text()
        if i!= 0 and ("SOMMAIRE" not in page) :
            text += (page or "") + "\n"

    with open("../data/doc.tkt", 'w') as f :
        f.write(text)

    return chunk_text(text)


if __name__ == "__main__" :
    try :
        doc = (sys.argv[1])
    except IndexError :
        print(f"Usage : {sys.argv[0]} fichier.pdf")
        sys.exit(1)
    except FileNotFoundError : 
        print(f"Fichier introuvable : {sys.argv[1]}")
        sys.exit(1)
    
    chunks = doc_chunked(doc)

    for i,chunk in enumerate(chunks) :
        if i>330 and i<335 : 
            print(f"Chunk {i} : {chunk}")