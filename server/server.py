from fastapi import FastAPI, Request
from typing import List
from pydantic import BaseModel
import spacy

nlp = spacy.load("mk_core_news_sm")

class RequestBody(BaseModel):
    sentences: List[str]
    keyword: str
    pos_category: str

app = FastAPI()
@app.post("/spacy_pos/")
async def spacy_pos(request: RequestBody):
    matching_sentences = []
    
    sentences = request.sentences
    keyword = request.keyword
    pos_category = request.pos_category
    # print(f'BODY-------------\n {sentences} \n--------------')
    for sentence in sentences:
        doc = nlp(sentence)
        for token in doc:
            if token.text.lower() == keyword.lower() and pos_category.lower() in spacy.explain(token.pos_).lower():
                matching_sentences.append(sentence)
                break
    return {"matching_sentences": matching_sentences}


@app.post("/macedonizer/")
async def macedonizer(request: RequestBody):
    pass


