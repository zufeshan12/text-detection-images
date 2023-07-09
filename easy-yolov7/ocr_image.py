from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
import cv2


yolov7 = YOLOv7()
yolov7.set(ocr_classes=['text'])
yolov7.load('best.weights', classes='classes.yaml',ocr_weights='best', device='cpu') # use 'gpu' for CUDA GPU inference
img = 'S1013797.JPG'
image_path = 'images_with_text/'+img
image = cv2.imread(img)

print("1")
detections = yolov7.detect(image)
print("2")
detected_image = draw(image, detections)
cv2.imwrite('detected_ocr.jpg', detected_image)
print(json.dumps(detections, indent=4))