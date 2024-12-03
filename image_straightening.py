import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os
import numpy as np
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import requests

from dotenv import load_dotenv
from ultralytics import YOLO
url = 'http://127.0.0.1:8000/extract_text'
path = "C:\\Users\\NICHOLAS\\Downloads\\KTP-CROP.v1i.yolov11\\train\\images"
results = []
i = 1
for filename in os.listdir(path):
    img = Image.open(path + "\\" + filename)
    print(filename)
    dst = np.array(img)
    i += 1
    print(i)

    with open(path + "\\" + filename, 'rb') as image_file:
        # Define the files dictionary to include the image
        files = {'file': (path + "\\" + filename, image_file, 'image/jpeg')}
        
        # Make the POST request with the image
        response = requests.post(url, files=files)
        print(response.content)
        with open("./new.json", "a") as outfile:
            print("a")
            json.dump({"contents": [{"file": filename, "result": str(response.content)}]}, outfile)
            outfile.write("\n")
          
