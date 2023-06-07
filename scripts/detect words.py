
from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
from matplotlib import pyplot as plt # plot images
import cv2 #opencv
import os # folder directory navigation

ocr_model = PaddleOCR(lang='en')

img_path = os.path.join('./data', 'invoice_sample.jpg')

# Run the ocr method on the ocr model
result = ocr_model.ocr(img_path)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./data/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('./data/result.jpg')