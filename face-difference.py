# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import cv2
import numpy as np

imagePath1 = sys.argv[1]
imagePath2 = sys.argv[2]




faceCascade = cv2.CascadeClassifier('/Users/davinclark/Downloads/haarcascade_frontalface_alt.xml')
net = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

imagePaths = np.array([imagePath1,imagePath2])
vectors = np.empty([2,128])
for i,imagePath in enumerate(imagePaths):
    image =  cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5
    )

    buffer  = 20
    if len(faces) == 0:
        print("BAD PHOTO")
        exit()
    (x, y, w, h) = faces[0]
    print("Face Found")
    
    face = image[int(y-(buffer*h/100)):int(y+h+(buffer*h/100)),
                    int(x-(buffer*w/100)):int(x+w+(buffer*w/100))]
    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(faceBlob)
    vec = net.forward()
    print("Face Vector: ")
    print(vec)
    print("\n")
    vectors[i] = vec
    print("processed image")


dist = np.linalg.norm(vectors[0]-vectors[1])
print(vectors[0],vectors[1])
print("distance is: ",dist)
if dist < 0.7:
    print("THESE ARE THE SAME FACES!!!!!")
else:
    print("THESE ARE DIFFERENT FACES!!!!!!")
