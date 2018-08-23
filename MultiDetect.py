import cv2
import numpy as np

mat=np.genfromtxt("train.txt",dtype=str,delimiter=';')
mat_x,mat_y=mat[:,0],mat[:,1]
train_x,train_y=[],[]
name=["YueBH","ZhaoRF","BaiQL"]
for i in mat_x:
    #print type(i)
    img=cv2.imread(i,0)
    train_x.append(img)
for j in mat_y:
    train_y.append(int(j))
model=cv2.face.createEigenFaceRecognizer()
model_1=cv2.face.createLBPHFaceRecognizer()
model.train(np.asarray(train_x),np.asarray(train_y))
model_1.train(np.asarray(train_x),np.asarray(train_y))
model.save('./model.xml')
model_1.save('model1.xml')
cap=cv2.VideoCapture(0)
kernel=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=kernel.detectMultiScale(gray,1.2,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,233,0),1)
        roi=gray[y:y+h,x:x+w]
        roi_new=cv2.resize(roi,(200,200),interpolation=cv2.INTER_LINEAR)
        parameters=model.predict(roi_new)
        parameters_1=model_1.predict(roi_new)
        if parameters[0]==parameters_1[0]:
            #print"Who r U:{} confidence: fisher{:.3f} LBP:{:.3f}".format(name[parameters[0]],parameters[1],parameters_1[1])
            print"Name :{}".format(name[parameters[0]])
            cv2.putText(frame,name[parameters[0]],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    cv2.imshow("who:",frame )
    if cv2.waitKey(30) & 0xff==ord("q"):
            break
cap.release()
cv2.destroyAllWindows()


