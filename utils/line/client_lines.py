#!/usr/bin/python

import numpy as np
import cv2
from pi2R.lines import *
from pi2R.point_cloud import *

name1 = '10000.jpg'
name2 = '10001.jpg'


line = Line(name1, name2)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',line.n_img)


cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.imshow('image2',line.l_img)




#np.set_printoptions(threshold=np.inf)
#a = np.argwhere(img_tt)

#print a, '!'
#a.view('int32,int32') #.sort(order=['f1'],axis=0)
rp = line.get_points_3d()
#print a
name = 'mask'
cv2.namedWindow(name, cv2.WINDOW_NORMAL)
cv2.imshow(name, line.mask)

img = cv2.bitwise_and(line.n_img, line.n_img, mask = line.mask)
#for i in rp:
#    cv2.circle(img, (i[0],i[1]), 1, (255,0,0), -1)

name = 'image2+points'
cv2.namedWindow(name, cv2.WINDOW_NORMAL)
cv2.imshow(name, img)

pc = PointCloud()
pc.addPoints(rp)

run(pc)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break



#cv2.waitKey(0)
cv2.destroyAllWindows()
