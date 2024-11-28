from PIL import Image
import json
import numpy as np
import cv2
from imutils.perspective import four_point_transform
from vertexai.generative_models import Part    
import re
import io
from google.cloud import storage
import os

class OCRService:
    def __init__(self, model_segment, model_fotokopi, model_genai):
        self.model_fotokopi = model_fotokopi
        self.model_segment = model_segment
        self.model_genai = model_genai

    def cvt_BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    def contrast(self,img):
        """
        Increase contrast of image
        
        Args:
            img : A Matlike or numpy array image
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe=cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

        lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l,a,b=cv2.split(lab)  # split on 3 different channels

        l2=clahe.apply(l)  # apply CLAHE to the L-channel

        lab=cv2.merge((l2,a,b))  # merge channels
        img2=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
        return img2

    def compute_focus_measure(self,image: np.ndarray) -> float:
        """
        Compute the focus measure of an image using the Laplacian variance method.
        A higher variance indicates a sharper (less blurry) image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def detect_blur_with_grid(self,image: np.ndarray, grid_size: int = 7) -> float:
        """
        Detect blur in an image by subdividing it into a grid, computing focus measures
        for each grid cell, and aggregating the results.
        
        Args:
        - image (np.ndarray): The input image.
        - grid_size (int): The number of subdivisions along each axis (e.g., 5x5 grid).

        Returns:
        - float: The aggregated focus measure for the full image.
        """
        h, w, _ = image.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        focus_values = []

        # Subdivide the image into grid cells and compute focus measure for each cell
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = image[y1:y2, x1:x2]
                focus_values.append(self.compute_focus_measure(cell))

        # Aggregate focus measures (e.g., using max, mean, or median)
        aggregated_focus = np.mean(focus_values)
        return aggregated_focus

    def blur_detection(self,image: np.ndarray, threshold: float, grid_size: int = 5) -> bool:
        """
        Determine if an image is blurry based on a threshold for the focus measure.

        Args:
        - image (np.ndarray): The input image.
        - threshold (float): The threshold below which the image is considered blurry.
        - grid_size (int): The number of subdivisions along each axis.

        Returns:
        - bool: True if the image is blurry, False otherwise.
        """
        focus_measure = self.detect_blur_with_grid(image, grid_size=grid_size)
        print(f"Focus Measure: {focus_measure}")
        if focus_measure < threshold:
            return "Blurry"
        return "Not Blurry"


    def flatten_image(self,array_img, mask_array):
        """
        Flattens an image to a quadrilateral based on the mask
        
        Args:
            array_img: Image in array format (Matlike or numpy array)
            mask_array: Matrix of segmentation mask by model
        """
        hull = cv2.convexHull(mask_array)

        # Approximate the convex hull to a quadrilateral
        epsilon = 0.02 * cv2.arcLength(hull, True)  # Adjust the epsilon for more or less approximation
        approx_quad = cv2.approxPolyDP(hull, epsilon, True)

        # If the approximation has more than 4 points, adjust or retry with a different epsilon
        if len(approx_quad) != 4:
            print("Failed to approximate to quadrilateral; current approximation has", len(approx_quad), "points.")
        else:
            print("a")
            array_img = four_point_transform(array_img, approx_quad.reshape(4, 2))
        return array_img
    
    def upload_to_gcs(self,file, destination_blob_name, bucket_name):
        """
        Uploads a file object to Google Cloud Storage.
        
        Args:
            file: The file object to upload.
            destination_blob_name: The name of the destination file in the bucket.
        """
        try:
            # Initialize Google Cloud Storage client
            img_byte_array = io.BytesIO()
            file.save(img_byte_array, format='PNG')  # Save image as PNG (can also be 'JPEG', 'BMP', etc.)
            img_byte_array.seek(0)  # Rewind the file pointer to the beginning
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob("ktp/"+destination_blob_name)
            
            # Upload file
            blob.upload_from_file(img_byte_array, content_type='image/png')
            return blob.public_url
        except Exception as e:
            return {"detail" : "Error upload to Google Storage. " + str(e)}
    
    def format_text(self,text):
        if (text[:7] == "```json"):
            text = text[8:len(text)-4]

        text = re.sub(r'(?<!")(\b[A-Za-z_]+\b)(?=\s*:)', r'"\1"', text)  # Fix unquoted keys
        text = re.sub(r'(?<=:\s)([A-Za-z0-9._\- ]+)(?=,|\n|\})', r'"\1"', text)  # Fix unquoted values

        return text
    
    def extract_text(self,file):
        img = Image.open(file.file)
        dst = np.array(img)    

        res_segment = self.model_segment.predict(source=img, save=False, task = "segment", show=False, conf=0.8)
    
        if (res_segment[0].masks == None):
            return {"detail":"Gambar bukan KTP"}
        
        res_fotokopi = self.model_fotokopi.predict(source=img, save=False, task = "classify", show=False, conf=0.8)
        if res_fotokopi[0].probs.top1 == 0:
            return {"detail":"Gambar merupakan fotokopi KTP"}

        masks = res_segment[0].masks.xy
        mask_array = np.array(masks, dtype=np.int32)
        mask_array = mask_array.reshape((-1, 1, 2))  # Reshape for OpenCV

        dst = self.flatten_image(dst,mask_array)
        
        dst = self.contrast(dst)

        if (self.blur_detection(dst,200) == "Blurry"):
            return {"detail":"Gambar blur, kirim ulang gambar"}
        
        dst = Image.fromarray(dst)

        gcs_filename = self.upload_to_gcs(dst, file.filename, os.getenv("BUCKET"))
        if isinstance(gcs_filename,dict):
            return gcs_filename

        #TODO ENV
        image_file = Part.from_uri(
           "gs://" + os.getenv("BUCKET") + "/ktp/" + file.filename, "image/jpeg"
        )   

        result = self.model_genai.generate_content(
        [image_file,os.getenv("PROMPT")]
        )
        text = result.text
        print(text)
        #Buat function formatting
        text = self.format_text(text)
        
        json_ktp = json.loads(text)

        if ("NIK" in json_ktp.keys() and len(json_ktp["NIK"]) != 16):
            json_ktp["message"] = "NIK bukan 16 angka"
        return json_ktp

    def perform_ocr(self, image):
        """Extract text from a KTP image."""
        try:
            return self.extract_text(image)
        except Exception as e:
            return {"detail" : str(e)}