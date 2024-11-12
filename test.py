import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os
import numpy as np
import cv2
from imutils.perspective import four_point_transform
import time

from dotenv import load_dotenv
from ultralytics import YOLO

model_segment = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\KTP_Segmentation.pt')
model_genai = genai.GenerativeModel("gemini-1.5-flash")

load_dotenv()

app = FastAPI()
API_KEY = os.getenv("API_KEY")
def extractText(file):    
    start = time.time()
    genai.configure(api_key=API_KEY)

    img = Image.open(file.file)
    dst = np.array(img)    

    res_segment = model_segment.predict(source=img, save=False, task = "segment", show=False, conf=0.8)
 
    if (res_segment[0].masks == None):
        return {"detail":"Gambar bukan KTP"}
    masks = res_segment[0].masks.xy
    mask_array = np.array(masks, dtype=np.int32)
    mask_array = mask_array.reshape((-1, 1, 2))  # Reshape for OpenCV

    hull = cv2.convexHull(mask_array)

    # Approximate the convex hull to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(hull, True)  # Adjust the epsilon for more or less approximation
    approx_quad = cv2.approxPolyDP(hull, epsilon, True)

    # If the approximation has more than 4 points, adjust or retry with a different epsilon
    if len(approx_quad) != 4:
        print("Failed to approximate to quadrilateral; current approximation has", len(approx_quad), "points.")
    else:
        dst = four_point_transform(dst, approx_quad.reshape(4, 2))
        
    output_path = 'C:\\OCR-KTP\\OCRR\\OCR-KTP'
    cv2.imwrite(output_path+'/test.jpg', dst)
    print(f"Image saved as {output_path}")

    dst = Image.fromarray(dst)
    
    result = model_genai.generate_content(
    [dst, "\n\n", """ 

    Ekstrak teks pada gambar dan NIK, Nama, Tanggal Lahir dan Alamat yang terdiri dari Alamat, RT/RW, Kelurahan/Desa dan Kecamatan ke dalam format JSON di bawah tanpa tambahan ```json```
        NIK hanya berjumlah 16 digit, tidak lebih dan tidak kurang, pastikan tidak melakukan output angka yang duplikat
        {
        "NIK": "0000000000000000",
        "Nama": "ABC",
        "Tanggal Lahir": "01-02-2000",
        "Alamat": {"Alamat": "ABC", "RT/RW": "001/003", "Kelurahan/Desa": "ABC", "Kecamatan": "ABC"}
        }
    """]
    )

    print(result.usage_metadata)
    billing = (result.usage_metadata.candidates_token_count*4800/1000000) + (result.usage_metadata.prompt_token_count*1200/1000000)
    json_ktp = json.loads(result.text)

    if None in json_ktp.values() or "null" in json_ktp.values(): 
        return {"detail":"Gambar tidak jelas"}

    if ("NIK" in json_ktp.keys() and len(json_ktp["NIK"]) != 16):
        json_ktp["message"] = "NIK bukan 16 angka"
    end = time.time()
    print(end - start)
    return json_ktp


@app.post("/extract_text/")
def extract_text_ktp(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    try:
        res = extractText(file)
        if "detail" in res.keys():
            raise HTTPException(status_code=400, detail=res["detail"])
        else:
            return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))