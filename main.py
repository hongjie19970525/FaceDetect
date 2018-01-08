'''
导入相关模块
'''
import os  
import numpy  
from PIL import Image,ImageDraw  
import cv2  
  
cap = cv2.VideoCapture(0)   #调用摄像头
fps = cap.get(cv2.CAP_PROP_FPS) #
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')#编码格式
#cv2.VideoWriter_fourcc('X','V','T','D')
#cv2.VideoWriter_fourcc('I','4','2','0')  
video = cv2.VideoWriter("aaa1.avi", fourcc, 5, size)#保存文件为output.avi,5帧 
print(cap.isOpened())  
  
classifier=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")   #调用训练文件  
  
count=0  
while count > -1:  
    ret,img = cap.read()  
    faceRects = classifier.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE,(20,20))  
    if len(faceRects)>0:  
        for faceRect in faceRects:   
                x, y, w, h = faceRect  
                cv2.rectangle(img, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), (0,255,0),2,0)  #在人脸上画矩形
    video.write(img)  
    cv2.imshow('video',img)  
    key=cv2.waitKey(1)  
    if key==ord('q'):  
        break                                            #当按下q键时，退出程序
  
video.release()  
cap.release()  
cv2.destroyAllWindows()  
