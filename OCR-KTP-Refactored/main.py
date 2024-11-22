import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os
import numpy as np
import cv2
from imutils.perspective import four_point_transform
from commands.ocr_command import OCRCommand
from services.ocr_service import OCRService

from dotenv import load_dotenv
from ultralytics import YOLO



model_segment = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\KTP_Segmentation.pt')
model_genai = genai.GenerativeModel("gemini-1.5-flash")

load_dotenv()

app = FastAPI()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)

ocr_service = OCRService(model_segment,model_genai)
ocr_command = OCRCommand(ocr_service=ocr_service)



@app.post("/extract_text/")
def extract_text_ktp(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    #try:
    res = ocr_command.execute(file)
    if "detail" in res.keys():
        raise HTTPException(status_code=400, detail=res["detail"])
    else:
        return res
    #except Exception as e:
    #    raise HTTPException(status_code=400, detail=str(e))