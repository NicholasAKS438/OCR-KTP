import google.generativeai as genai
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os
import numpy as np
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\NICHOLAS\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

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
    
    res_detect = model_detect.predict(source=img, save=False, task = "detect", show=False, conf=0.8)
    if len(res_detect[0].boxes.conf) != 1:
      raise HTTPException(status_code=400, detail="Gambar bukanlah KTP")

    res_segment = model_segment.predict(source=img, save=False, task = "segment", show=False, conf=0.8)
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
        # Unpack the points if the approximation succeeded in yielding a quadrilateral
        quad_points = approx_quad.reshape(-1, 2)
        quad_points = np.array(quad_points, dtype=np.float32)
        print("Approximated Quadrilateral Points:", quad_points)

        width_top = np.linalg.norm(quad_points[1] - quad_points[0])
        width_bottom = np.linalg.norm(quad_points[2] - quad_points[3])
        height_left = np.linalg.norm(quad_points[3] - quad_points[0])
        height_right = np.linalg.norm(quad_points[2] - quad_points[1])

        # Use the maximum of the widths and heights as the side length for the square

        square_size = int(max(width_top, width_bottom, height_left, height_right))
        print(square_size)
        # Define the destination points to form a square
        square_pts = np.float32([
            [0, 0],                 # Top-left corner
            [400, 0],   # Top-right corner
            [400, 611],  # Bottom-right corner
            [0, 611]    # Bottom-left corner
        ])

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(quad_points, square_pts)

        dst = cv2.warpPerspective(img_array, M, (400,611))
    
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, adjusted_image)
    print(f"Image saved as {output_path}")

    dst = Image.fromarray(dst)
    result = model_genai.generate_content(
        [dst, "\n\n", """ 
        
        Ekstrak teks pada gambar dan NIK, Nama, Tanggal Lahir dan Alamat yang terdiri dari Alamat, RT/RW, Kelurahan/Desa dan Kecamatan ke dalam format JSON di bawah tanpa tambahan ```json```
         Usahakan untuk tetap mengekstrak teks yang ada, terutama nama dan tanggal lahir
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

    if "NIK" in json_ktp.keys() and len(json_ktp["NIK"]) != 16:
      print("a")
      res = model_genai.generate_content(
        [img, "\n\n", """NIK is always 16 digits, write only the NIK out without any extra text"""]
      )
      print(res.text)
      json_ktp["NIK"] = res.text.strip()
      
    
       

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
