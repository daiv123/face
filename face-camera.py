# -*- coding: utf-8 -*-


import sys
import cv2



faceCascade = cv2.CascadeClassifier('/Users/davinclark/Downloads/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

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
            imageSize = 200
            scale = imageSize/image.shape[1]
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            dim = (height, width)
            print(scale)
            print(width)
            # resize image
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow('face',cv2.flip(image,1))


    cv2.imshow('capture', cv2.flip(img,1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



