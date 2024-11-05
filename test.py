from ultralytics import YOLO
model = YOLO('OCRR\\OCR-KTP\\best.pt')  # load a pretrained YOLO detection model
res = model("C:\\Users\\NICHOLAS\\Downloads\\NOT_KTP\\yl6auqxlw4tchjmb92sr.jpg")

print(res[0].boxes)