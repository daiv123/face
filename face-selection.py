import sys
import os
import cv2

inDir = sys.argv[1]
outDir = sys.argv[2]
if len(sys.argv) > 3:
    i = int(sys.argv[3])
else :
    i = 0


faceCascade = cv2.CascadeClassifier('/Users/davinclark/Downloads/haarcascade_frontalface_alt.xml')
faceString = "face{}.png"
images = [cv2.imread(file.path) for file in os.scandir(inDir) if (file.is_file() and file.name[0] != '.')]

os.chdir(outDir)
for img in images:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1)
    for (x,y,w,h) in faces:
        imgCopy = img.imgCopy()
        imgCopy = cv2.rectangle(imgCopy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("press k to keep",imgCopy)
        key = 0xFF & cv2.waitKey()
        if key == ord('k'):
            cv2.imwrite(faceString.format(i), img[y:y+h, x:x+w])
            i+=1
        cv2.destroyAllWindows()

    
            
