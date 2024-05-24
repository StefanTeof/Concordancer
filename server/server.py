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
import sys
from mysql.connector import Error

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from database.db_connection import connect_to_database


nlp = spacy.load("mk_core_news_sm")

tokenizer = RobertaTokenizer.from_pretrained('macedonizer/mk-roberta-base')
model = RobertaModel.from_pretrained('macedonizer/mk-roberta-base')
svm_model = joblib.load('../Models/svm_model.pkl')

class SearchRequestBody(BaseModel):
    keyword: str
    pos_category: str

class SimpleSearchRequestBody(BaseModel):
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


def process_file(file_content, file_name: str):
    # If file_content is a byte-like object, decode it to a string
    if isinstance(file_content, bytes):
        text = file_content.decode("utf-8")
    else:
        # If it's already a string, no need to decode
        text = file_content
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    processed_data = []
    for i, sentence in enumerate(sentences):
        processed_data.append({
            "file_name": file_name,
            "sentence_id": i + 1,
            "sentence": sentence
        })
    
    return processed_data



@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    connection = connect_to_database()
    cursor = connection.cursor()
    
    file_content = await file.read()

    try:
        # Insert file details into the database
        cursor.execute(
            "INSERT INTO files (resource, context) VALUES (%s, %s)",
            (file.filename, file_content)
        )
        connection.commit()

        # Retrieve the ID of the inserted record
        file_id = cursor.lastrowid

    except Error as e:
        connection.rollback()  # Roll back in case of error
        print("Error while inserting file into MySQL", e)
        raise HTTPException(status_code=500, detail="File upload failed")

    finally:
        cursor.close()
        connection.close()

    return {"file_id": file_id, "file_name": file.filename}

@app.post("/simple_search/")
async def simple_search(request_body: SimpleSearchRequestBody):
    keyword = request_body.keyword.lower()

    if not keyword:
        raise HTTPException(status_code=400, detail="Keyword must be provided.")  
    
    lemmas = pd.read_csv('../lemmas.csv')
    
    related_words = set()
    related_words.add(keyword)
    if keyword in lemmas['word'].values:
        lemma = lemmas[lemmas['word'] == keyword]['lemma'].iloc[0]
        related_words.update(set(lemmas[lemmas['lemma'] == lemma]['word'].tolist()))
        related_words.add(lemma)
        
    connection = connect_to_database()
    cursor = connection.cursor()
    
    try:
        cursor.execute("SELECT file_id, resource, context FROM files")
        files = cursor.fetchall()
        search_results = []
        for file_id, file_name, file_content in files:
            content_lower = file_content.lower()
            for word in related_words:
                pattern = r'\b' + re.escape(word) + r'\b'
                for match in re.finditer(pattern, content_lower):
                    start = match.start()
                    end = match.end()
                    
                    # Define the context window size
                    window_size = 60
                    left_context = file_content[max(0, start - window_size):start]
                    right_context = file_content[end:min(len(file_content), end + window_size)]

                    search_results.append((file_name, left_context, word, right_context))

        return search_results

    except Error as e:
        print("Error while fetching files from MySQL", e)
        raise HTTPException(status_code=500, detail="Search failed")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


@app.post("/search/")
async def search(request_body: SearchRequestBody):
    keyword = request_body.keyword.lower()
    category = request_body.pos_category.lower()

    if not keyword or not category:
        raise HTTPException(status_code=400, detail="Both keyword and category must be provided.")

    lemmas = pd.read_csv('../lemmas.csv')
    
    related_words = set()
    related_words.add(keyword)
    if keyword in lemmas['word'].values:
        lemma = lemmas[lemmas['word'] == keyword]['lemma'].iloc[0]
        related_words.update(set(lemmas[lemmas['lemma'] == lemma]['word'].tolist()))
        related_words.add(lemma)
    

    connection = connect_to_database()
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT file_id, resource, context FROM files")
        files = cursor.fetchall()
        search_results = []
        
        for file_id, file_name, file_content in files:
            for word in related_words:
                spacy_results = filter_sentences_with_spacy(file_name, file_content, word, category)
                search_results.extend(spacy_results)

        # print(f"RESULTS ================ {search_results}")
        return JSONResponse(content={"results": search_results})

    except Error as e:
        print("Error while fetching files from MySQL", e)
        raise HTTPException(status_code=500, detail="Search failed")

    finally:
        cursor.close()
        connection.close()
        
        
def filter_sentences_with_spacy(file_name, text, keyword, pos_category):
    results = []
    doc = nlp(text)  # Process the whole text

    for token in doc:  # Iterate over tokens in the document
        if token.text.lower() == keyword.lower() and pos_category.lower() in spacy.explain(token.pos_).lower():
            start_index = token.idx
            end_index = start_index + len(token.text)

            # Extracting context more dynamically
            left_context = text[max(0, start_index - 60):start_index].rsplit(' ', 1)[0]
            right_context = text[end_index:min(len(text), end_index + 60)].split(' ', 1)[-1]

            results.append({
                "file_name": file_name,
                "left_context": left_context,
                "keyword": token.text,
                "right_context": right_context
            })

    return results


def filter_sentences_with_macedonizer(sentences_dict, keyword, pos_category):
    
    # sentence_to_filename = {entry["sentence"]: entry["file_name"] for entry in sentences_dict}
    
    sentences = [entry["sentence"] for entry in sentences_dict]

    # Example usage:
    cleaned_sentences = [remove_special_characters(sentence) for sentence in sentences]
    # print(f'CLEANED SENTENCES: {cleaned_sentences}')
    # Process sentences to obtain embeddings data
    embeddings_data = process_sentences(cleaned_sentences)
    # print(f'EMBEDDINGS DATA:  {embeddings_data}')
    # Create a DataFrame from embeddings data
    df = pd.DataFrame(embeddings_data)
    # print(df)
    # Convert embeddings to dataframe columns
    df = create_feature_embeddings(df, ['sentence_embedding', 'word_embedding'])
    df.drop(columns=['sentence_embedding', 'word_embedding'], inplace=True)
    # print(df)
    # Prepare the input for the model
    X = df.drop(columns=['sentence_id', 'word'])
    predicted_categories_indices = svm_model.predict(X)
    categories = ['0', 'adjective', 'adposition', 'adverb', 'conjuction', 'noun', 'numeral', 'particle', 'pronoun', 'residual', 'verb']
    predicted_categories = [categories[index] for index in predicted_categories_indices]
    df['predicted_category'] = predicted_categories

    # Map sentences to their words and predicted categories
    final_mapping_df = df[['sentence_id', 'word', 'predicted_category']]
        
    sentence_word_category_mapping = final_mapping_df.groupby('sentence_id').apply(lambda x: dict(zip(x.word, x.predicted_category))).to_dict()

    # Find sentences with the specified keyword and category
    result = find_sentences_with_keyword_and_category(keyword, pos_category, sentence_word_category_mapping, cleaned_sentences)
    
    final_result = []
    seen = set()  # Set to track seen (sentence, filename) pairs
    for s in result:
        for s2 in sentences_dict:
            tmp_s = remove_special_characters(s2['sentence'])
            if s == tmp_s:
                pair = (s, s2['file_name'])  # Create a tuple of the sentence and filename
                if pair not in seen:  # Check if the pair has not been added yet
                    seen.add(pair)  # Mark this pair as seen
                    final_result.append({"sentence": s, "file_name": s2['file_name']})
    
                
    print(f'FINAL RESULT: {final_result}')
    # result_with_filenames = [(sentence, sentence_to_filename[sentence]) for sentence in result if sentence in sentence_to_filename]
    
    return final_result
