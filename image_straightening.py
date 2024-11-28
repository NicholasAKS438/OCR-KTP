import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os
import numpy as np
import cv2
import numpy as np
from imutils.perspective import four_point_transform

from dotenv import load_dotenv
from ultralytics import YOLO
model_segment = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\src\\KTP_Fotokopi.pt')

path = "C:\\Users\\NICHOLAS\\Downloads\\KTP\\ktp"
results = []
i = 1
for filename in os.listdir(path):
    img = Image.open(path + "\\" + filename)
    print(filename)
    dst = np.array(img)
    i += 1
    print(i)


    res_segment = model_segment.predict(source=img, save=False, task = "classify", show=False, conf=0.8)
    
    if res_segment[0].probs.top1 == 1:
        pass

    else:
        img = np.array(img)
        img = Image.fromarray(img)
        output_path = "C:\\Users\\NICHOLAS\\Downloads\\ktp-fotokopi.v2i.folder\\train\\non" +filename
        img.save(output_path)
        print(f"Image saved as {output_path}")
print(results)