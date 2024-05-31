from fastapi import FastAPI
from typing import Annotated
from pydantic import BaseModel
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


app = FastAPI(
    title="Predict Book Genre",
    summary="This tool takes a book description and analyzes its content to determine the book's genre.")




class Book(BaseModel):
    description: str 
    
class Prediction(BaseModel):
    results: str


stop_words = stopwords.words('english')

model = joblib.load("models/book_genre_prediction_xg.joblib")
vectorizer = joblib.load("encoder_vectorizer/tfidf_vectorizer.joblib")
encoder = joblib.load("encoder_vectorizer/encoder.joblib")

def preprocess_text(text: str) -> str:
    """
    

    Parameters
    ----------
    text : str
        DESCRIPTION.

    Returns
    -------
    str
        DESCRIPTION.

    """
    text = re.sub("[^A-Za-z]", " ", text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stop_words)]
    text = " ".join(text)
    return text

text = """A dystopian novel set in a totalitarian society ruled by the Party and its leader, Big Brother. The protagonist, Winston Smith, struggles with oppression, surveillance, and the loss of individuality. Orwell's work is a powerful warning about the dangers of totalitarianism."""
pre = preprocess_text(text)
vec = vectorizer.transform([pre])
encoder.inverse_transform(model.predict(vec))[0]

@app.post("/predict/", response_model=Prediction)
async def give_description(description: str):
    text = preprocess_text(description)
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    decode_the_results = encoder.inverse_transform(prediction)[0]
    return {"results":decode_the_results}