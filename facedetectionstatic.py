import cv2
import os
import numpy as np

front_cascade = cv2.CascadeClassifier('D:\\faceproject\\haarcascade_frontalface_default.xml')

img = cv2.imread('D:\\faceproject\\WIN_20181130_10_13_43_Pro.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = front_cascade.detectMultiScale(gray, 1.3, 6)
for (x, y, w, h) in faces:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 155), 3)

print (faces)

cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows
