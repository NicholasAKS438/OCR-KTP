import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
API_KEY = os.getenv("API_KEY")
def extractText(file):
    


    genai.configure(api_key=API_KEY)

    img = Image.open(file.file)

    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content(
        [img, "\n\n", """Apa kelurahan pada teks, usahakan untuk mencocokkan dengan desa di Indonesia"""]
    )


    return result.text

@app.post("/extract_text/")
def extract_text_ktp(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
  
    return extractText(file)
