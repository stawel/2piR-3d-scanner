#!/usr/bin/python

import numpy as np
import cv2

name1 = '10001.jpg'
name2 = '10002.jpg'
def nothing(x):
    pass

# Load an color image in grayscale
img1 = cv2.imread(name1,cv2.IMREAD_COLOR)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('T','image',0,255,nothing)
cv2.imshow('image',img1)


img2 = cv2.imread(name2,cv2.IMREAD_COLOR)

cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.createTrackbar('T','image2',0,255,nothing)
cv2.imshow('image2',img2)



b1,g1,r1 = cv2.split(img1)
b2,g2,r2 = cv2.split(img2)

def add_window(a,b,name ,blur,  tr):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    img = cv2.absdiff(a,b)

    def d(t):
        if blur:
            img2 = cv2.GaussianBlur(img,(5,5),0)
        else:
            img2 = img
        r, img_t = cv2.threshold(img2, t, 255, tr) #)
        cv2.imshow(name,img_t)

    cv2.createTrackbar('T',name,0,255,d)
    cv2.imshow(name ,img)


add_window(g2,g1,'g',True, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
add_window(r2,r1,'r',True, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
add_window(b2,b1,'b',True, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

rgb2 = b2>>2
rgb2 +=r2>>2
rgb2 +=g2>>2
rgb1 = b1>>2
rgb1 +=r1>>2
rgb1 +=g1>>2


add_window(rgb2,rgb1,'rgb',False, cv2.THRESH_BINARY) #cv2.THRESH_TOZERO)
add_window(rgb2,rgb1,'rgb2',False, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #cv2.THRESH_TOZERO)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

#    t = cv2.getTrackbarPos('T','b')
#    r, img_t = cv2.threshold(img5, t, 0, cv2.THRESH_TOZERO)
#    cv2.imshow('b',img_t)


#cv2.waitKey(0)
cv2.destroyAllWindows()

