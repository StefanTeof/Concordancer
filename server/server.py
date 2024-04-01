from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import spacy
import joblib
from transformers import RobertaTokenizer, RobertaModel
import torch
import re
import numpy as np
import pandas as pd
import io


nlp = spacy.load("mk_core_news_sm")

tokenizer = RobertaTokenizer.from_pretrained('macedonizer/mk-roberta-base')
model = RobertaModel.from_pretrained('macedonizer/mk-roberta-base')
svm_model = joblib.load('../Models/svm_model.pkl')

class RequestBody(BaseModel):
    sentences: List[str]
    keyword: str
    pos_category: str
    
class SearchRequestBody(BaseModel):
    sentences_db: List[dict]
    keyword: str

app = FastAPI()

def remove_special_characters(text: str) -> str:
    pattern = r'[?.,!:;@#$%^&*()\[\]{}\\/|+\-_=]'
    return re.sub(pattern, '', text)

def process_sentences(sentences):
    embeddings_data = []
    sentence_id = 0  # Initialize sentence ID

    for sentence in sentences:
        cleaned_sentence = remove_special_characters(sentence)
        inputs = tokenizer(cleaned_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
        all_token_embeddings = outputs.last_hidden_state.squeeze().numpy()
        words = cleaned_sentence.split()

        token_idx = 1  # Skip [CLS]

        for word in words:
            token_embeddings = []
            while token_idx < len(all_token_embeddings) - 1:  # Skip [SEP]
                token_embedding = all_token_embeddings[token_idx]
                token_text = tokenizer.decode(inputs['input_ids'][0, token_idx], clean_up_tokenization_spaces=True).strip()

                if token_text == word or word.startswith(token_text):
                    token_embeddings.append(token_embedding)
                    token_idx += 1
                else:
                    break

            if token_embeddings:
                word_embedding = np.mean(token_embeddings, axis=0).tolist()
                embeddings_data.append({
                    'sentence_id': sentence_id,
                    'word': word,
                    'sentence_embedding': sentence_embedding,
                    'word_embedding': word_embedding
                })

        sentence_id += 1  # Increment sentence ID for the next sentence

    return embeddings_data


def create_feature_embeddings(df, columns):
    for column in columns:
        embedding_cols = pd.DataFrame(df[column].tolist(), columns=[f'{column}_{i}' for i in range(len(df[column][0]))])

        # Concatenate the new columns with the original dataframe
        df = pd.concat([df, embedding_cols], axis=1)
        
    return df

def find_sentences_with_keyword_and_category(keyword, category, sentence_word_category_mapping, sentences):
    matching_sentences = []
    keyword_lower = keyword.lower()
    category_lower = category.lower()

    # Iterate over the sentence ID and word-category mapping
    for sentence_id, word_category_map in sentence_word_category_mapping.items():
        # Check if the keyword exists in the current sentence's word-category map and matches the category, case insensitively
        for word, word_category in word_category_map.items():
            if word.lower() == keyword_lower and word_category.lower() == category_lower:
                # If a match is found, append the corresponding sentence to the matching_sentences list
                matching_sentences.append(sentences[sentence_id])
                break

    return matching_sentences

# Dict for storing the sentences temporary
db = {}

@app.post("/process_file/")
async def process_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Reset db and populate with new sentences
    db.clear()
    for i, sentence in enumerate(sentences):
        db[i + 1] = {'sentence': sentence}
    
    sentences_list = [{'sentence_id': sid, 'sentence': data['sentence']} for sid, data in db.items()]
    
    return JSONResponse(content={"sentences": sentences_list})


@app.post("/search/")
async def search(request_body: SearchRequestBody):
    sentences_db = request_body.sentences_db
    keyword = request_body.keyword.lower()
    matching_sentences = [item for item in sentences_db if keyword in item['sentence'].lower()]
    
    return JSONResponse(content={"matching_sentences": matching_sentences})

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


@app.post("/macedonizer_pos")
async def macedonizer_pos(request: RequestBody):
    # Example usage:
    cleaned_sentences = [remove_special_characters(sentence) for sentence in request.sentences]
    
    print(cleaned_sentences)
    embeddings_data = process_sentences(cleaned_sentences)

    # Continue with your data preparation and model prediction as before
    df = pd.DataFrame(embeddings_data)
    
    print(df)
    
    # Convert embeddings to dataframe columns
    df = create_feature_embeddings(df, ['sentence_embedding', 'word_embedding'])
    df.drop(columns=['sentence_embedding', 'word_embedding'], inplace=True)

    # Prepare the input for the model
    X = df.drop(columns=['sentence_id', 'word'])
    predicted_categories_indices = svm_model.predict(X)
    categories = ['0', 'adjective', 'adposition', 'adverb', 'conjuction', 'noun', 'numeral', 'particle', 'pronoun', 'residual', 'verb']
    predicted_categories = [categories[index] for index in predicted_categories_indices]
    df['predicted_category'] = predicted_categories

    # Map sentences to their words and predicted categories
    final_mapping_df = df[['sentence_id', 'word', 'predicted_category']]
    
    print(final_mapping_df)
        
    sentence_word_category_mapping = final_mapping_df.groupby('sentence_id').apply(lambda x: dict(zip(x.word, x.predicted_category))).to_dict()

    print(sentence_word_category_mapping)

    # Find sentences with the specified keyword and category
    result = find_sentences_with_keyword_and_category(request.keyword, request.pos_category, sentence_word_category_mapping, cleaned_sentences)
    
    return {"matching_sentences": result}

