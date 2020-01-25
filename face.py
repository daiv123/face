# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import cv2

imagePath = sys.argv[1]




faceCascade = cv2.CascadeClassifier('/Users/davinclark/Downloads/haarcascade_frontalface_alt.xml')
img = cv2.imread(imagePath)


scale_percent = 100 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.5
)

buffer  = 20
print(faces)
for (x, y, w, h) in faces:
    print("face found")
#    cropImage = image[int(y-(buffer*h/100)):int(y+h+(buffer*h/100)),
#                      int(x-(buffer*w/100)):int(x+w+(buffer*w/100))]
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow('faces found', image)
cv2.waitKey(0)
