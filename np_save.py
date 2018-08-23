import cv2
import numpy as np

mat=np.genfromtxt("train.txt",dtype=str,delimiter=';')
mat_x,mat_y=mat[:,0],mat[:,1]
train_x,train_y=[],[]
name=["YueBH","ZhaoRF","BaiQL"]
for i in mat_x:
    #print type(i)
    img=cv2.imread(i,0)
    #img_1=np.float32(img)
    train_x.append(img)
for j in mat_y:
    train_y.append(int(j))
#train_data=np.asarray(train_x)
#train_label=np.asarray(train_y)
#print train_x,train_y.shape
np.save("train_data",train_x)
np.save("train_labels",train_y)
'''

a=np.load("train_data.npy")
b=np.load("train_labels.npy")
c=b.T
d=a.T
print a.shape,b.shape,c.shape,d.shape
#print a.size(),b.size()

model=cv2.ml.SVM_create()
model.train(np.asarray(train_x),np.asarray(train_y))
a=np.load('tain.txt')
a[0]'''