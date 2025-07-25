# Import necessary libraries
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

from transformers import MarianMTModel, MarianTokenizer
from typing import List
from fastapi.responses import JSONResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the MarianMT model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-roa"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Define a request schema
class TranslationRequest(BaseModel):
    src_text: List[str]

# Define a response schema
class TranslationResponse(BaseModel):
    translated_text: List[str]

@app.get("/")
def read_root():
    return {"message": "Welcome to the MarianMT Translation API!"}

@app.post("/translate", response_model=TranslationResponse)
def translate_text(request: TranslationRequest):
    try:
        # Tokenize the input text
        inputs = tokenizer(request.src_text, return_tensors="pt", padding=True)
        # Generate translations
        translated = model.generate(**inputs)
        # Decode the translations
        translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return TranslationResponse(translated_text=translated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
