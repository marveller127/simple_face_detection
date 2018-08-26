import cv2
import numpy as np
face_detect=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
count=200
while(cap.isOpened()):
    ret,frame=cap.read(0)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_detect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,234),3)
        f=cv2.resize(gray[y:y+h,x:x+w],(200,200))
        #cv2.imwrite("./fir_rawdata_pgm/%s.pgm" %str(count),f)
        cv2.imwrite("bql/{}.jpg".format(count), f)
        count +=1
        print ("%s" %str(count))
        if count==300:
            cap.release()
            cv2.destroyAllWindows()
            break
    cv2.imshow("camera:",frame)
    cv2.waitKey(40)


