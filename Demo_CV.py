import cv2
import numpy as np
import time
load_from_disk = True
if load_from_disk:
    hsv_value = np.load('penval.npy') #
print (hsv_value)
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
kernel = np.ones((5,5),np.uint8)
canvas = None
x1,y1=0,0
noiseth = 800
while(1):
    _, frame = cap.read() # Take each frame
    frame = cv2.flip( frame, 1 ) 
    if canvas is None: #จอสีดำที่ขึ้นข้อความ
        canvas = np.zeros_like(frame) #return array of 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #covert RGB to HSV
    if load_from_disk: 
        lower_range = hsv_value[0]
        upper_range = hsv_value[1]
    else:           
        lower_range  = np.array([80, 132, 159])
        upper_range = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_range, upper_range) #maskจุดที่สีตรง
    mask = cv2.erode(mask,kernel,iterations = 1) #ลด noise
    mask = cv2.dilate(mask,kernel,iterations = 2) #ลด noise
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and cv2.contourArea(max(contours,key = cv2.contourArea)) > noiseth:
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c) #อ่านจุดจาก contours
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2     
        else:
            canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 4) #วาดเส้น
        x1,y1= x2,y2 #กำหนดจุดเริ่มใหม่เป็นจุดจบ เพื่อให้เส้นมันต่อ ๆกัน
       
    else:
        x1,y1 =0,0
    frame = cv2.add(frame,canvas)
    stacked = np.hstack((canvas,frame))
    cv2.imshow('VIRTUAL PEN',cv2.resize(stacked,None,fx=0.6,fy=0.6))

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k == ord('c'):
        canvas = None

cv2.destroyAllWindows()
cap.release()