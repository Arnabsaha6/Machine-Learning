# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:12:26 2019

@author: arnab saha
"""
import cv2
#import numpy as np
cascadepath="haarcascade_frontalface_default.xml"
facedetector=cv2.CascadeClassifier(cascadepath)
cam=cv2.VideoCapture(0)
id=input("Enter ID")
sNo=0
while(True):
    rect,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetector.detectMultiScale(gray,1.1,5)
    for(x,y,w,h) in faces:
        sNo=sNo+1
        cv2.imwrite("data/user."+str(id)+"."+str(sNo)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)
    cv2.imshow("face",img)
    cv2.waitKey(1)
    if sNo>100:
        break
cam.release()
cv2.destroyAllWindows()