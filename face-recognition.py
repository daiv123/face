import cv2
import pickle
import sys
from sklearn import svm

faceCascade = cv2.CascadeClassifier('/Users/davinclark/Downloads/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

(model, label_dict) = pickle.load(open(sys.argv[1], "rb"))

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    disp = img.copy()
    disp = cv2.flip(disp,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5
    )
    for (x,y,w,h) in faces :
        face = img[y:y+h, x:x+w]
        
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(faceBlob)
        vec = net.forward()
        prediction = model.predict(vec)

        disp = cv2.rectangle(disp, (len(img[0])-x,y), (len(img[0])-(x+w), y+h), (0,255,0))
        cv2.putText(disp, label_dict[prediction[0]], (len(img[0])-x-w, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imshow('capture', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

