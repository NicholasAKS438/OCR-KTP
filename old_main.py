import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os
import numpy as np
import cv2
from imutils.perspective import four_point_transform
from vertexai.generative_models import GenerativeModel, Part
from vertexai.tuning import sft
import os
import re

from dotenv import load_dotenv
from ultralytics import YOLO

model_segment = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\KTP_Segmentation.pt')
model_genai = genai.GenerativeModel("gemini-1.5-flash")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\OCR-KTP\OCRR\OCR-KTP\credentials.json'
sft_tuning_job = sft.SupervisedTuningJob("projects/67912531469/locations/us-central1/tuningJobs/6663904508663300096")
tuned_model = GenerativeModel(sft_tuning_job.tuned_model_endpoint_name)

load_dotenv()

app = FastAPI()
API_KEY = os.getenv("API_KEY")


def cvt_BGR2RGB(img):
  return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def contrast(img):
  # CLAHE (Contrast Limited Adaptive Histogram Equalization)
  clahe=cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

  lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
  l,a,b=cv2.split(lab)  # split on 3 different channels

  l2=clahe.apply(l)  # apply CLAHE to the L-channel

  lab=cv2.merge((l2,a,b))  # merge channels
  img2=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
  return img2

def blur_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < 300:
        return "Blurry"
    return "Not Blurry"


def flatten_image(array_img, mask_array):
    hull = cv2.convexHull(mask_array)

    # Approximate the convex hull to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(hull, True)  # Adjust the epsilon for more or less approximation
    approx_quad = cv2.approxPolyDP(hull, epsilon, True)

    # If the approximation has more than 4 points, adjust or retry with a different epsilon
    if len(approx_quad) != 4:
        print("Failed to approximate to quadrilateral; current approximation has", len(approx_quad), "points.")
    else:
        array_img = four_point_transform(array_img, approx_quad.reshape(4, 2))
    return array_img

def extractText(file):
    genai.configure(api_key=API_KEY)

    img = Image.open(file.file)
    dst = np.array(img)    

    res_segment = model_segment.predict(source=img, save=False, task = "segment", show=False, conf=0.8)
 
    if (res_segment[0].masks == None):
        return {"detail":"Gambar bukan KTP"}
    masks = res_segment[0].masks.xy
    mask_array = np.array(masks, dtype=np.int32)
    mask_array = mask_array.reshape((-1, 1, 2))  # Reshape for OpenCV

    dst = flatten_image(dst,mask_array)
    
    dst = contrast(dst)

    if (blur_detection(dst) == "Blurry"):
        return {"detail":"Gambar blur, kirim ulang gambar"}
    
    dst = Image.fromarray(dst)
    image_file = Part.from_uri(
    "gs://ktp-stash/ktp/test.jpg", "image/jpeg"
    )   
    result = tuned_model.generate_content(
    [image_file,""" 
    Ekstrak teks pada gambar dan identifikasi NIK, Nama, Tanggal Lahir dan Alamat yang terdiri dari Alamat, RT/RW, Kelurahan/Desa dan Kecamatan ke dalam format JSON seperti di bawah tanpa tambahan teks dan ```json```
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
    print(text)

    if (result.text[:7] == "```json"):
        text = text[8:len(text)-4]
    text = re.sub(r'(?<!")(\b[A-Za-z_]+\b)(?=\s*:)', r'"\1"', text)  # Fix unquoted keys
    text = re.sub(r'(?<=:\s)([A-Za-z0-9._\- ]+)(?=,|\n|\})', r'"\1"', text)  # Fix unquoted values
    json_ktp = json.loads(text.strip("\n"))

    #NOTE Remove if not every value has to be filled
    if None in json_ktp.values() or "null" in json_ktp.values() or None in json_ktp['Alamat'].values():  
        return {"detail":"Gambar tidak jelas"}

    if ("NIK" in json_ktp.keys() and len(json_ktp["NIK"]) != 16):
        json_ktp["message"] = "NIK bukan 16 angka"
    return json_ktp

@app.post("/extract_text/")
def extract_text_ktp(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    #try:
    res = extractText(file)
    if "detail" in res.keys():
        raise HTTPException(status_code=400, detail=res["detail"])
    else:
        return res
    #except Exception as e:
    #    raise HTTPException(status_code=400, detail=str(e))