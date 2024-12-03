from doctr.io import DocumentFile
# PDF
from doctr.models import ocr_predictor
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

from doctr.models import kie_predictor

# Model
model = kie_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

single_img_doc = DocumentFile.from_images("C:\\OCR-KTP\\OCRR\\OCR-KTP\\test.jpg")

result = model(single_img_doc)
print(result)