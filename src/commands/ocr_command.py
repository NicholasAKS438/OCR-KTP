from .base import Command
from services.ocr_service import OCRService

class OCRCommand(Command):
    def __init__(self, ocr_service: OCRService):
        self.ocr_service = ocr_service

    def execute(self, image):
        """Execute OCR on the given image."""
        return self.ocr_service.perform_ocr(image)