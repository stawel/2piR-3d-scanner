#!/usr/bin/python


import numpy as np
import cv2
import math

from scipy.signal import argrelmax, argrelmin
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from timeit import default_timer as timer


def bright_mask(h1,s1,v1,h2,s2,v2):
    # bright objects
    mask_bright1 = 0.9 < v1
    mask_bright2 = 0.9 < v2

    mask_satur = s2+0.05 < s1
    return np.logical_and(np.logical_and(mask_bright1, mask_bright2),mask_satur)

def max_laser(y, h1,v1,s1,h2,v2,s2, not_mask):
    start = timer()
    y[not_mask] = -1.
#    y = gaussian_filter1d(y, sigma=2, axis=1)
    y = gaussian_filter(y, sigma=2)
    y[not_mask] = -1.
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

def wrap_red(h):
    h[h>0.5]-=1.
    return h

def split(img):
    return wrap_red(img[:,:,0]),img[:,:,1],img[:,:,2]

def transform(img1, img2):
    start = timer()
    h1,s1,v1 = split(img1)
    h2,s2,v2 = split(img2)

    y = (v2-v1)**2 + (s2-s1)**2+(h2-h1)**2

    mask_brighter = v2 > v1+0.07

#    mask_b = bright_mask(h1,s1,v1, h2,s2,v2)
#    mask_red = red_mask(h1,s1,v1, h2,s2,v2)

#    mask = np.logical_and(np.logical_or(mask_b,mask_brighter), mask_red)
#    mask_not = np.logical_not(mask)
    mask_not = np.logical_not(mask_brighter)

    x,y = max_laser(y, h1,v1,s1,h2, v2, s2, mask_not)

    end = timer()
    print 'transform time:',end - start

    return mask_not, (x,y)

def get_points_2d_g(img1, img2):
    start = timer()
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV);
    img1_hsv = img1_hsv.astype(np.float32)/[180.,255.,255.]

    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    img2_hsv = img2_hsv.astype(np.float32)/[180.,255.,255.]
    image, (x,y) = transform(img1_hsv, img2_hsv)
    end = timer()
    print 'get_points_2d_g time:',end - start
    return x,y
