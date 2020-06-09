# -*- coding: utf-8 -*-


import sys
import cv2
import serial
import math
import time

delay = 1
past = time.time()

faceCascade = cv2.CascadeClassifier('/Users/davinclark/Downloads/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")
ser = serial.Serial('/dev/cu.usbmodem14201', 9600)

ser.write(("90"+"\n").encode())

while(True):
    
        # Capture frame-by-frame
    ret, img = cap.read()

    # resize image
    
    image = img.copy()
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5
    )
    
    buffer  = 20
    if len(faces) != 0:
        (x, y, w, h) = faces[0]
    
        image = image[int(y-(buffer*h/100)):int(y+h+(buffer*h/100)),
                        int(x-(buffer*w/100)):int(x+w+(buffer*w/100))]
    
        

        if image.size > 10 and image.size < img.size:
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 5)
            if(past + delay < time.time()):
                past = time.time()
                ser.write((str(135-(math.floor((x+(w/2))*90/img.shape[1])))+"\n").encode())
                time.sleep(0.05)
                print(time.time()-past)
            



    cv2.imshow('capture', cv2.flip(img,1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



