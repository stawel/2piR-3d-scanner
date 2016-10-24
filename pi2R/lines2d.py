#!/usr/bin/python


import numpy as np
import cv2
import math
import scipy.ndimage as ndimage

from scipy.signal import argrelmax, argrelmin
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from timeit import default_timer as timer


def bright_mask(h1,s1,v1,h2,s2,v2):
    # bright objects
    mask_bright1 = 0.9 < v1
    mask_bright2 = 0.9 < v2

    mask_satur = s2+0.05 < s1
    return np.logical_and(np.logical_and(mask_bright1, mask_bright2),mask_satur)

def max_laser(y, d, not_mask):
    start = timer()
    y[not_mask] = d
#    y = gaussian_filter1d(y, sigma=2, axis=1)
#    y = gaussian_filter(y, sigma=2)
#    y[not_mask] = d
    retu = argrelmax(y, axis=1)
    end = timer()
    print 'max_laser time:',end - start

    return retu

def red_mask(h1,s1,v1,h2,s2,v2):
    # bright objects
    d = 0.1
    mask_red1 = h2 < 2*d
    mask_red2 = -d > h2
    return np.logical_or(mask_red1, mask_red2)

def split(img):
    return img[:,:,0],img[:,:,1],img[:,:,2]

y_data = []
def transform(img1, img2):
    global y_data
    start = timer()
    h1,l1,s1 = split(img1)
    h2,l2,s2 = split(img2)

#    l1 = ndimage.gaussian_filter1d(l1, 3, axis=1)
#    l2 = ndimage.gaussian_filter1d(l2, 3, axis=1)
#    l1 = ndimage.gaussian_filter1d(l1, 3, axis=0)
#    l2 = ndimage.gaussian_filter1d(l2, 3, axis=0)
#    s1 = ndimage.gaussian_filter1d(l1, 3, axis=1)
#    s2 = ndimage.gaussian_filter1d(l2, 3, axis=1)
    y_data = l2-l1#+(s2-s1)*0.3
#    w = l2 > 0.4
#    y_data[w] += (s2-s1)[w]
    d = 0.00
#    mask_brighter = np.logical_and(y > d, l2 > 0.15)
    mask_brighter = y_data > d
    (sy,sx,sz) = img2.shape
    mask_brighter[sy-10:sy,:] = False

    mask_not = np.logical_not(mask_brighter)

    x,y = max_laser(y_data, d, mask_not)

    end = timer()
    print 'transform time:',end - start

    return mask_not, (x,y)

def get_points_2d_g(img1, img2):
    start = timer()
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS);
    img1_hsv = img1_hsv.astype(np.float32)/[180.,255.,255.]

    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)
    img2_hsv = img2_hsv.astype(np.float32)/[180.,255.,255.]
    image, (x,y) = transform(img1_hsv, img2_hsv)
    end = timer()
    print 'get_points_2d_g time:',end - start
    return x,y
