import cv2
import sys
import os
import numpy as np
import pickle

dir = sys.argv[1]
dest = sys.argv[2]
label = sys.argv[3]
net = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")


images = [cv2.imread(file.path) for file in os.scandir(dir) if (file.is_file() and file.name[0] != '.')]

embeddings = []

for image in images :


    faceBlob = cv2.dnn.blobFromImage(image, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(faceBlob)
    vec = net.forward()
    embeddings.append((vec.tolist(), label))
pickle.dump(embeddings, open(dest, "wb"))
