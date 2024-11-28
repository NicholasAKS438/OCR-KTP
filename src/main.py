from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from commands.ocr_command import OCRCommand
from services.ocr_service import OCRService

from vertexai.generative_models import GenerativeModel
from vertexai.tuning import sft
from dotenv import load_dotenv
from ultralytics import YOLO



model_segment = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\src\\KTP_Segmentation.pt')
model_fotokopi = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\src\\KTP_Fotokopi.pt')

load_dotenv()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\OCR-KTP\OCRR\OCR-KTP\src\credentials.json'
sft_tuning_job = sft.SupervisedTuningJob(os.getenv("TUNING_JOB"))
tuned_model = GenerativeModel(sft_tuning_job.tuned_model_endpoint_name)

app = FastAPI()
API_KEY = os.getenv("API_KEY")

ocr_service = OCRService(model_segment, model_fotokopi,tuned_model)
ocr_command = OCRCommand(ocr_service=ocr_service)

@app.post("/extract_text/")
def extract_text_ktp(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    try:
        res = ocr_command.execute(file)
        if "detail" in res.keys():
            raise HTTPException(status_code=400, detail=res["detail"])
        else:
            return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))