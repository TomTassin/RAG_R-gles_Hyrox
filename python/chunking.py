from typing import TypedDict

import pymupdf4llm
import pymupdf
import numpy as np
import re
import sys
from pathlib import Path
import json

PAGES_SOMMAIRE = 3

TITLE_SIZES = {"chapter": 40, "section": 16, "subsection": 10}
TOLERANCE_SIZE = 0.5

LEFT_COLUMN_MAX_X = 280

class Chunk(TypedDict):
    chapter: str
    section: str
    page: int
    text: str
    source: str #name of the providence file


def chunk_text (chapter: str, section : str, text : str, page : int, source : str, chunk_size : int = 40, overlap : int = 20) -> list[Chunk]:
    """ 
    chunk_size  : number of words per chunk
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if section :
        header = f"{chapter} > {section}" 
    else :
        header = chapter
        section = chapter

    words = text.split()
    chunks: list[Chunk] = []
    start = 0

    while start < len(words) : 
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start : end]).strip()
        if chunk:
            chunks.append({
                "chapter": chapter,
                "section": section,
                "page": page,
                "text": f"{header}\n{chunk}",
                "source": source
            })

        if end == len(words):          # fin atteinte : pas de chunk en double
            break
        
        start += chunk_size - overlap

    return chunks

def level_size(size : int) -> bool :
    for level in TITLE_SIZES :
        if abs(size - TITLE_SIZES[level]) <= TOLERANCE_SIZE :
            return level
    return None

def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def box_text(box: dict) -> str:
    return clean(" ".join(span['text'] for line in box['textlines'] or [] for span in line['spans']))


def box_titles(box: dict) -> tuple[str, str]:
    chapter, section = "", ""
    for line in box['textlines'] or []:
        if line['bbox'][0] >= LEFT_COLUMN_MAX_X:
            continue
        for span in line['spans']:
            level = level_size(span['size'])
            if level == "chapter":
                chapter += " " + span['text']
            elif level is not None:
                section += " " + span['text']
    return clean(chapter), clean(section)


def table_chunk(chapter: str, section: str, box: dict, page: int, source: str) -> Chunk:
    if not section : 
        section = chapter 
    return {
        'chapter': chapter,
        'section': section,
        'page': page,
        'text': box['table']['markdown'],
        'source': source,
    }


def box_chunked(data: dict, source_file: str) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    chapter = ""
    section = ""
    for page in data['pages'][PAGES_SOMMAIRE:]:
        page_number = page['page_number']
        for box in page['boxes']:
            if box['boxclass'] in ('page-footer', 'picture', 'caption'):
                continue

            if box['boxclass'] not in ('text', 'table'):
                new_chapter, new_section = box_titles(box)
                if new_chapter:                      # nouveau chapitre : on vide la section
                    chapter, section = new_chapter, ""
                if new_section:
                    section = new_section
                if new_chapter or new_section:       # c'était une boîte de titre
                    continue

            if box['boxclass'] == 'table' and box['table']:
                all_chunks.append(table_chunk(chapter, section, box, page_number, source_file))
                continue

            text = box_text(box)
            if len(text) < 10 :
                continue
            all_chunks.extend(chunk_text(chapter=chapter, section=section, text=text,
                                         page=page_number, source=source_file))
    return all_chunks


def load_doc(pdf : str) :
    path_pdf = Path(pdf)
    if not path_pdf.exists() : 
        print(f"File : {pdf} not found")
        sys.exit(1)
    
    doc = pymupdf.open(path_pdf)
    doc_json = doc.to_json(use_ocr=False)
    
    return json.loads(doc_json)

def doc_chunked(pdf : str) -> list[Chunk] :
    data = load_doc(pdf)
    return box_chunked(data, pdf)

if __name__ == "__main__" :
    try :
        pdf = (sys.argv[1])
    except :
        print(f"Usage : {sys.argv[0]} file.pdf")
        sys.exit(1)

    data = load_doc(pdf)
    final_chunks = doc_chunked(pdf)
    for i in range (10) : 
        print(f"CHUNK {i} : {final_chunks[i]}")
        print()
