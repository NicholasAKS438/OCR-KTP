from PIL import Image
import json
import numpy as np
import cv2
from imutils.perspective import four_point_transform

class OCRService:
    def __init__(self, model_segment, model_genai):
        self.model_segment = model_segment
        self.model_genai = model_genai

    def cvt_BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    def contrast(self,img):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe=cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

        lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l,a,b=cv2.split(lab)  # split on 3 different channels

        l2=clahe.apply(l)  # apply CLAHE to the L-channel

        lab=cv2.merge((l2,a,b))  # merge channels
        img2=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
        return img2

    def blur_detection(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm < 300:
            return "Blurry"
        return "Not Blurry"


    def flatten_image(self,array_img, mask_array):
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

    def extract_text(self,file):
        img = Image.open(file.file)
        dst = np.array(img)    

        res_segment = self.model_segment.predict(source=img, save=False, task = "segment", show=False, conf=0.8)
    
        if (res_segment[0].masks == None):
            return {"detail":"Gambar bukan KTP"}
        masks = res_segment[0].masks.xy
        mask_array = np.array(masks, dtype=np.int32)
        mask_array = mask_array.reshape((-1, 1, 2))  # Reshape for OpenCV

        dst = self.flatten_image(dst,mask_array)
        
        dst = self.contrast(dst)

        if (self.blur_detection(dst) == "Blurry"):
            return {"detail":"Gambar blur, kirim ulang gambar"}
        
        dst = Image.fromarray(dst)

        result = self.model_genai.generate_content(
        [dst,""" 
        Ekstrak teks pada gambar dan identifikasi NIK, Nama, Tanggal Lahir dan Alamat yang terdiri dari Alamat, RT/RW, Kelurahan/Desa dan Kecamatan ke dalam format JSON seperti di bawah tanpa tambahan teks dan ```json```
            Tempat lahir tidak termasuk dalam tanggal lahir
            Berikan null jika informasi teks blur atau susah diekstrak
            NIK hanya berjumlah 16 digit, tidak lebih dan tidak kurang, pastikan tidak melakukan output angka yang duplikat
            {
            "NIK": "0000000000000000",
            "Nama": "ABC",
            "Tanggal Lahir": "01-02-2000",
            "Alamat": {"Alamat": "ABC", "RT/RW": "001/003", "Kelurahan/Desa": "ABC", "Kecamatan": "ABC"}
            }
        """]
        )
        text = result.text
        print(text)

        if (result.text[:7] == "```json"):
            text = text[8:len(text)-4]

        json_ktp = json.loads(text)

        #NOTE Remove if not every value has to be filled
        if None in json_ktp.values() or "null" in json_ktp.values() or None in json_ktp['Alamat'].values():  
            return {"detail":"Gambar tidak jelas"}

        if ("NIK" in json_ktp.keys() and len(json_ktp["NIK"]) != 16):
            json_ktp["message"] = "NIK bukan 16 angka"
        return json_ktp

    def perform_ocr(self, image):
        """Extract text from an image."""
        try:
            return self.extract_text(image)
        except Exception as e:
            return {"detail" : str(e)}