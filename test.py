import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os
import numpy as np
import cv2
from imutils.perspective import four_point_transform

from dotenv import load_dotenv
from ultralytics import YOLO
model_detect = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\KTP_Detection.pt')  # load a pretrained YOLO detection model
model_segment = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\best.pt')
model_genai = genai.GenerativeModel("gemini-1.5-flash")

load_dotenv()

app = FastAPI()
API_KEY = os.getenv("API_KEY")
def extractText(file):    
    genai.configure(api_key=API_KEY)

    img = Image.open(file.file)
    img_array = np.array(img)    
    """"
    res_detect = model_detect.predict(source=img, save=False, task = "detect", show=False, conf=0.8)
    if len(res_detect[0].boxes.conf) != 1:
        raise HTTPException(status_code=400, detail="Gambar bukanlah KTP")

    res_segment = model_segment.predict(source=img, save=False, task = "segment", show=False, conf=0.8)
    masks = res_segment[0].masks.xy
    

    mask_array = np.array(masks, dtype=np.int32)
    mask_array = mask_array.reshape((-1, 1, 2))  # Reshape for OpenCV
    """""

    # Load image, grayscale, Gaussian blur, Otsu's threshold

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours and sort for largest contour
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    for c in cnts:
        # Perform contour approximation
        epsilon = 0.02 * cv2.arcLength(c, True)
        step = epsilon * 0.5
        
        while True:
            # Approximate the contour
            approx = cv2.approxPolyDP(c, epsilon, True)
            
            # Check if the approximation has exactly four points
            if len(approx) == 4:
                break
            elif len(approx) < 4:
                # If fewer than 4 points, decrease epsilon to make it less precise
                epsilon -= step
            else:
                # If more than 4 points, increase epsilon to make it more precise
                epsilon += step
        print(approx)
        if len(approx) == 4:
            displayCnt = approx
            break

    # Obtain birds' eye view of image
    warped = four_point_transform(img_array, displayCnt.reshape(4, 2))



    output_path = 'C:\\OCR-KTP\\OCRR\\OCR-KTP'

    cv2.imwrite(output_path+'/thresh.jpg', thresh)
    cv2.imwrite(output_path+"/warped.jpg", warped)
    print(f"Image saved as {output_path}")
    dst = Image.fromarray(warped)

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

    print(result.text)

    json_ktp = json.loads(result.text)

    print(json_ktp)

    if None in json_ktp.values() or "null" in json_ktp.values(): 
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
     #   raise HTTPException(status_code=400, detail=str(e))
