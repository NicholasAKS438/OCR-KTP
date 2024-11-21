import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os
import numpy as np
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import json

from dotenv import load_dotenv
from ultralytics import YOLO

results = []
model_genai = genai.GenerativeModel("gemini-1.5-flash")

load_dotenv()

app = FastAPI()
API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)
for filename in os.listdir(".\\ktp"):
    img = Image.open(".\\ktp\\" + filename)

    result = model_genai.generate_content(
    [img, "\n\n", """ 
    Ekstrak teks pada gambar dan identifikasi NIK, Nama, Tanggal Lahir dan Alamat yang terdiri dari Alamat, RT/RW, Kelurahan/Desa dan Kecamatan ke dalam format JSON seperti di bawah tanpa tambahan ```json```
        Tempat lahir tidak termasuk dalam tanggal lahir
        Berikan null jika informasi teks blur atau susah diekstrak
        NIK hanya berjumlah 16 digit, tidak lebih dan tidak kurang, pastikan tidak melakukan output angka yang duplikat
        {
        "NIK": "0000000000000000",
        "Nama": "ABC",
        "Tanggal Lahir": "01-02-2000",
        "Alamat": {"Alamat": "ABC", "RT/RW": "001/003", "Kelurahan/Desa": "ABC", "Kecamatan": "ABC"}
        }
    """]
    )
    text = result.text
    if (result.text[:7] == "```json"):
        text = text[8:len(text)-4]
    with open("sample.json", "a") as outfile:
        print("a")
        json.dump({"contents": [{"role": "user", "parts": [{"fileData": {"mimeType": "image/jpeg", "fileUri": f"gs://gemini-tuning-438/ktp/{filename}"}}, {"text": """Ekstrak teks pada gambar dan identifikasi NIK, Nama, Tanggal Lahir dan Alamat yang terdiri dari Alamat, RT/RW, Kelurahan/Desa dan Kecamatan ke dalam format JSON seperti di bawah tanpa tambahan ```json```
        Tempat lahir tidak termasuk dalam tanggal lahir
        Berikan null jika informasi teks blur atau susah diekstrak
        NIK hanya berjumlah 16 digit, tidak lebih dan tidak kurang, pastikan tidak melakukan output angka yang duplikat
        {
        "NIK": "0000000000000000",
        "Nama": "ABC",
        "Tanggal Lahir": "01-02-2000",
        "Alamat": {"Alamat": "ABC", "RT/RW": "001/003", "Kelurahan/Desa": "ABC", "Kecamatan": "ABC"}
        }"""}]}, {"role": "model", "parts": [{"text": text}]}]}, outfile)
        outfile.write('\n')
        

    

    
 
