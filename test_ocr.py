from PIL import Image, ImageFilter
import cv2
import blur_detector
# Open the input image
image = Image.open('./test.jpg')

face_cascade = cv2.CascadeClassifier('cv2/data/haarcascade_frontalface_default.xml')

img = cv2.imread('./test.jpg', 0)
print(img)
blur_map1 = blur_detector.detectBlur(img, downsampling_factor=1, num_scales=3, scale_start=1)
print(blur_map1)