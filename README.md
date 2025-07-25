# Linguaflow

Linguaflow is a multilingual translation API built using FastAPI and MarianMT from Hugging Face Transformers. This project enables translation between English and other languages like French, Portuguese, and Spanish. The project was developed as part of a course at **Grow Data Skills**.

---

## Features
- Translate text from English to other languages.
- Supported languages: French, Portuguese, Spanish.
- Simple and efficient API.

---

## Prerequisites
- Python 3.8 or later
- pip

---

## Setup and Installation

Follow these steps to set up and run the project:

### 1. Clone the Repository
```bash
# Clone the repository to your local machine
git clone <repository-url>
cd linguaflow
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test the API
Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) to view the root endpoint. Use the `/translate` endpoint to translate text.

#### Sample Payload:
```json
{
  "src_text": [">>esp<< This is a test sentence to translate to Spanish."]
}
```

#### Sample Response:
```json
{
  "translated_text": ["Esta es una frase de prueba para traducir al español."]
}
```

---

## Project Structure
```
Linguaflow/
│
├── app/
│   ├── main.py         # FastAPI application code
│
├── requirements.txt    # Dependency list
├── README.md           # Project documentation
```

---

## Developed As Part of Grow Data Skills
This project was created as part of a hands-on project in the **Grow Data Skills** course. The course focuses on practical applications of machine learning and natural language processing.

---


