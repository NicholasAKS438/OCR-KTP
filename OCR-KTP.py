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
        [img, "\n\n", """Extract the text in image in an JSON and NIK, Name, Date of Birth and Address Which consists of Alamat, RT/RW, Kelurahan/Desa and Kecamatan from array into JSON with this format with the newlines and without any additional text and without ```json``` and always try to fill every line
        {
          "NIK": "0000000000000000",
          "Name": "ABC",
          "Date of Birth": "01-02-2000",
          "Alamat": {"Alamat":"ABC", "RT/RW": "001/003", "Kelurahan/Desa": "ABC", "Kecamatan": "ABC"}
          }
        Give only this JSON if information is not clear
        {"message":"Gambar tidak jelas"}
        """]
    )

    json_ktp = json.loads(result.text)
    print(json_ktp)

    return json_ktp

@app.post("/extract_text/")
def extract_text_ktp(file: UploadFile = File(...)):
  if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
  try:
    return extractText(file)
  except Exception as e:
    raise HTTPException(status_code=500, detail="Error processing image") from e