import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from commands.ocr_command import OCRCommand
from services.ocr_service import OCRService

from vertexai.generative_models import GenerativeModel, Part
from vertexai.tuning import sft
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
genai.configure(api_key=API_KEY)

ocr_service = OCRService(model_segment,tuned_model)
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