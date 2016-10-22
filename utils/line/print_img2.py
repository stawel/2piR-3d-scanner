#!/usr/bin/python

import numpy as np
import cv2

name1 = '10334.jpg'
name2 = '10335.jpg'
def nothing(x):
    pass

def rotate(img):
    #return cv2.flip(cv2.transpose(img),0)
    return img

# Load an color image in grayscale
img1 = cv2.imread(name1,cv2.IMREAD_COLOR)
img1 = rotate(img1)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('T','image',0,255,nothing)
cv2.imshow('image',img1)


img2 = cv2.imread(name2,cv2.IMREAD_COLOR)
img2 = rotate(img2)

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


add_window(g2,g1,'g: cv2.THRESH_BINARY',True, cv2.THRESH_BINARY)
add_window(r2,r1,'r: cv2.THRESH_BINARY',True, cv2.THRESH_BINARY)
add_window(b2,b1,'b: cv2.THRESH_BINARY',True, cv2.THRESH_BINARY)

rgb2 = b2>>1
rgb2 +=r2>>1
#rgb2 +=g2>>2
rgb1 = b1>>1
rgb1 +=r1>>1
#rgb1 +=g1>>2


add_window(rgb2,rgb1,'rgb: cv2.THRESH_BINARY',False, cv2.THRESH_BINARY) #cv2.THRESH_TOZERO)
add_window(rgb2,rgb1,'rgb: cv2.THRESH_BINARY+cv2.THRESH_OTSU',False, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #cv2.THRESH_TOZERO)



img = cv2.absdiff(rgb2,rgb1)
r, img_t = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_tt = img_t

def reduce_pix(a):
    retu = []
    cy = -1
    cx = -1
    cx2 = 0
    for i in a:
        x = i[1]
        y = i[0]
        if cy != y:
            if cy >= 0:
#                print cx,cy
                retu.append([(cx+cx2)/2,cy])

            cy = y
            cx = x
            cx2 = x
        elif x < cx2+5:
            cx2 = x
        else:
#            print '!', cx,cy
            retu.append([(cx+cx2)/2,cy])
            cx2 = x
            cx = x
    return retu

np.set_printoptions(threshold=np.inf)
a = np.argwhere(img_tt)

#print a, '!'
#a.view('int32,int32') #.sort(order=['f1'],axis=0)
rp = reduce_pix(a)
#print a
name = 'points, rgb: cv2.THRESH_BINARY+cv2.THRESH_OTSU'
cv2.namedWindow(name, cv2.WINDOW_NORMAL)
cv2.imshow(name ,img_tt)

img2 = cv2.bitwise_not(img2,img2,mask = img_tt)
for i in rp:
    cv2.circle(img2, (i[0],i[1]), 1, (255,0,0), -1)

name = 'image2+points'
cv2.namedWindow(name, cv2.WINDOW_NORMAL)
cv2.imshow(name,img2)



while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

#    t = cv2.getTrackbarPos('T','b')
#    r, img_t = cv2.threshold(img5, t, 0, cv2.THRESH_TOZERO)
#    cv2.imshow('b',img_t)


#cv2.waitKey(0)
cv2.destroyAllWindows()

