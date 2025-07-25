from fastapi import FastAPI, Request
from pydantic import BaseModel
import gradio as gr
import requests
import threading
import uvicorn

# Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-roa"
headers = {"Authorization": "Bearer YOUR_HUGGINGFACE_API_KEY"}  # Replace with your token

app = FastAPI()

# ----------------- FASTAPI SECTION --------------------
class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
def translate_api(req: TranslationRequest):
    payload = {"inputs": req.text}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return {"error": response.text}

    try:
        translated = response.json()[0]['translation_text']
        return {"translated_text": translated}
    except Exception as e:
        return {"error": str(e)}

# ----------------- GRADIO SECTION ---------------------
def translate_gradio(text):
    if not text.strip():
        return "Please enter some text."
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    try:
        return response.json()[0]['translation_text']
    except Exception:
        return f"Unexpected response: {response.json()}"

def launch_gradio():
    demo = gr.Interface(
        fn=translate_gradio,
        inputs=gr.Textbox(lines=4, placeholder="Enter English text..."),
        outputs="text",
        title="English to Romance Language Translator",
        description="Translates English to Romance languages using MarianMT via Hugging Face API."
    )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# Start Gradio in a separate thread
threading.Thread(target=launch_gradio).start()

# ----------------- ENTRY POINT ------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
