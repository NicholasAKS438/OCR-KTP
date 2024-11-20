from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv4', use_space_char=True) # need to run only once to download and load model into memory
img_path = 'C:\\OCR-KTP\\OCRR\\OCR-KTP\\ktp\\ktpIMG-20231013-WA0139_jpg.rf.d1e356cc93b99c1d7969198d7a9f2bda.jpg'
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)


# draw result
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
print("line"+str(line[0][1][0]))
txts = [line[1][0] for line in result[0]]
for line in result[0]:
    print("line" + str(line))
print(str(txts))