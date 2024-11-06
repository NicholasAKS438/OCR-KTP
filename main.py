import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np

from dotenv import load_dotenv
from ultralytics import YOLO
model_detect = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\KTP_Detection.pt')  # load a pretrained YOLO detection model
model_segment = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\KTP_Segmentation.pt')
model_genai = genai.GenerativeModel("gemini-1.5-flash")


load_dotenv()

app = FastAPI()
API_KEY = os.getenv("API_KEY")
def extractText(file):    
    genai.configure(api_key=API_KEY)

    img = Image.open(file.file)

    res_detect = model_detect.predict(source=img, save=False, task = "detect", show=False, conf=0.8)
    print(res_detect[0].boxes)
    if len(res_detect[0].boxes.conf) == 0:
      raise HTTPException(status_code=400, detail="Gambar bukanlah KTP")

    res_segment = model_segment.predict(source=img, save=False, task = "segment", show=False, conf=0.8)
    print(res_segment[0].masks.xy)
    masks = res_segment[0].masks.xy

    mask_array = np.array(masks, dtype=np.int32)
    mask_array = mask_array.reshape((-1, 1, 2))  # Reshape for OpenCV

    result = model_genai.generate_content(
        [img, "\n\n", """ 
        
        Ekstrak teks pada gambar dan NIK, Nama, Tanggal Lahir dan Alamat yang terdiri dari Alamat, RT/RW, Kelurahan/Desa dan Kecamatan ke dalam format JSON dan tanpa teks tambahan dan tanpa ```json```
         Alamat adalah teks di bawah jenis kelamin dan golongan darah dan di atas RT/RW, jangan mengambil dari Tempat/Tanggal lahir
         Usahakan untuk mencocokan alamat dengan desa atau kecamatan yang ada di Indonesia yang terdekat
         Alamat disusun sesuai dengan urutan berikut pada gambar:
          Alamat, RT/RW, Kelurahan/Desa, Kecamatan
         {
          "NIK": "0000000000000000",
          "Nama": "ABC",
          "Tanggal Lahir": "01-02-2000",
          "Alamat": {"Alamat": "ABC", "RT/RW": "001/003", "Kelurahan/Desa": "ABC", "Kecamatan": "ABC"}
          }
        Berikan hanya JSON ini jika gambarnya buram
        {"message": "Gambar buram"}
        Berikan hanya JSON ini jika gambar tersebut bukan KTP
        {"message": "Gambar bukan KTP"}
        """]
    )
    
        
    json_ktp = json.loads(result.text)
    
    print(json_ktp)

    if "NIK" in json_ktp.keys() and len(json_ktp["NIK"]) != 16:
      print("a")
      res = model_genai.generate_content(
        [img, "\n\n", """NIK is always 16 digits, write only the NIK out without any extra text"""]
      )
      print(res.text)
      json_ktp["NIK"] = res.text.strip()
      
    
    if "NIK" in json_ktp.keys() and json_ktp["NIK"][0] != "1":
      json_ktp["NIK"] = "1" + json_ktp["NIK"][1:]
      print(json_ktp["NIK"])
       

    if None in json_ktp.values() or ("NIK" in json_ktp.keys() and len(json_ktp["NIK"]) != 16): 
        return {"message": "Gambar tidak jelas"}

    return json_ktp

@app.post("/extract_text/")
def extract_text_ktp(file: UploadFile = File(...)):
  if not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="File is not an image")
  #try:
  return extractText(file)
  #except Exception as e:
  #  raise HTTPException(status_code=400, detail="Image processing failed")
