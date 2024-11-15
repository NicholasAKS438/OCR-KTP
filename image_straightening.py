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
model_segment = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\best (5).pt')

def contrast(img):
  # CLAHE (Contrast Limited Adaptive Histogram Equalization)
  clahe=cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

  lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
  l,a,b=cv2.split(lab)  # split on 3 different channels

  l2=clahe.apply(l)  # apply CLAHE to the L-channel

  lab=cv2.merge((l2,a,b))  # merge channels
  img2=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
  return img2

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

results = []
for filename in os.listdir("C:\\Users\\NICHOLAS\\Downloads\\KTP\\ktp"):
    img = Image.open("C:\\Users\\NICHOLAS\\Downloads\\KTP\\ktp\\" + filename)
    print(filename)
    dst = np.array(img)

    

    res_segment = model_segment.predict(source=img, save=False, task = "segment", show=False, conf=0.8)
    
    if res_segment[0].masks != None:
        masks = res_segment[0].masks.xy
        mask_array = np.array(masks, dtype=np.int32)
        mask_array = mask_array.reshape((-1, 1, 2))  # Reshape for OpenCV

        dst = flatten_image(dst,mask_array)
    
        dst = contrast(dst)

        output_path = "C:\\OCR-KTP\\OCRR\\OCR-KTP\\ktp\\ktp" +filename
        cv2.imwrite(output_path, dst)
        print(f"Image saved as {output_path}")

    else:
        results.append(filename)
print(results)