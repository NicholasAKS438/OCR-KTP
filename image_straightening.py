import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
import json
import os
import numpy as np
import cv2
import numpy as np

from dotenv import load_dotenv
from ultralytics import YOLO
model_detect = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\KTP_Detection.pt')  # load a pretrained YOLO detection model
model_segment = YOLO('C:\\OCR-KTP\\OCRR\\OCR-KTP\\best.pt')


results = []
for filename in os.listdir("C:\\Users\\NICHOLAS\\Downloads\\KTP\\ktp"):
    img = Image.open("C:\\Users\\NICHOLAS\\Downloads\\KTP\\ktp\\" + filename)
    print(filename)
    img_array = np.array(img)

    

    res_segment = model_segment.predict(source=img, save=False, task = "segment", show=False, conf=0.8)
    

    if res_segment[0].masks != None:
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

        output_path = "C:\\OCR-KTP\\OCRR\\OCR-KTP\\ktp\\ktp" +filename
        cv2.imwrite(output_path, dst)
        print(f"Image saved as {output_path}")

    else:
        results.append(filename)
print(results)