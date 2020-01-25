import sys
import os
import cv2

inDir = sys.argv[1]
outDir = sys.argv[2]


faceCascade = cv2.CascadeClassifier('/Users/davinclark/Downloads/haarcascade_frontalface_alt.xml')
faceString = "face{}.png"
for entry in os.scandir(inDir):
    if(entry.is_file() and entry.name[0]!="."):
        img = cv2.imread(entry.name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.5)
        os.chdir(outDir)
        for i, (x,y,w,h) in enumerate(faces):
            imgCopy = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("press k to keep",imgCopy)
            key = 0xFF & cv2.waitKey()
            if key == ord('k'):
                cv2.imsave(faceString.format(i))
            cv2.destroyAllWindows()
    
            
