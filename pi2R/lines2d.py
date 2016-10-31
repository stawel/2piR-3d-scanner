#!/usr/bin/python


import numpy as np
import cv2
import math
import scipy.ndimage as ndimage
import scipy.signal as signal


from scipy.signal import argrelmax, argrelmin
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from timeit import default_timer as timer


def max_laser(y):
    start = timer()

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
kernel_size_x = 15
kernel_size_y = 15
kt = 1.8
kernel = np.outer(signal.gaussian(kernel_size_y,kt,),signal.gaussian(kernel_size_x,kt,))
kernel.astype(np.float32)

def transform(img1, img2, t=3):
    global kernel, y_data
    start = timer()
    h1,l1,s1 = split(img1)
    h2,l2,s2 = split(img2)

    y_t_data = l2-l1
#    dy = 0.0
#    y_data[y_data<dy] = -0.2

    y_t_data = signal.fftconvolve(y_t_data, kernel, mode='same')

#    ymax = y_data.max()
#    y_data[500:500+kernel_size_y,500:500+kernel_size_x] = ymax*kernel
#    print 'kernel sum:', kernel.sum(), 'ymax:', ymax
    #d = ymax/10
#    print y_data
#    d = 0.5
    d=0.4
    mask_not = y_t_data < d
#    mask_not[:,0:20] = True
#    mask_not[:,-20:] = True
#    mask_not[0:20,:] = True
#    mask_not[-20:,:] = True
    y_t_data[mask_not] = d
    x,y = max_laser(y_t_data)

#    y_data = (l2-l1)

    y_data = y_t_data

    end = timer()
    print 'transform time:',end - start

    return mask_not, (x,y)

def get_points_2d_g(img1, img2):
    start = timer()
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)
    image, (x,y) = transform(img1_hsv, img2_hsv)
    end = timer()
    print 'get_points_2d_g time:',end - start
    return x,y
