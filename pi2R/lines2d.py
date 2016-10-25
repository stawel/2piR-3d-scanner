#!/usr/bin/python


import numpy as np
import cv2
import math
import scipy.ndimage as ndimage
import scipy.signal as signal


from scipy.signal import argrelmax, argrelmin
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from timeit import default_timer as timer


def bright_mask(h1,s1,v1,h2,s2,v2):
    # bright objects
    mask_bright1 = 0.9 < v1
    mask_bright2 = 0.9 < v2

    mask_satur = s2+0.05 < s1
    return np.logical_and(np.logical_and(mask_bright1, mask_bright2),mask_satur)

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

def transform(img1, img2, t=25):
    global y_data, kernel
    start = timer()
    h1,l1,s1 = split(img1)
    h2,l2,s2 = split(img2)

    y_data = l2-l1
    dy = 0.03
    y_data[y_data<dy] = dy

    kernel_size_x = 25
    kernel_size_y = 11
    kernel_x = np.ones((kernel_size_x), dtype=np.float32)*-1
    kernel_x[kernel_size_x/4:(kernel_size_x*3)/4] = 1.
    kernel = np.outer(signal.gaussian(kernel_size_y,7,),kernel_x)
#    kernel/= kernel.sum()#kernel_size_y*kernel_size_x*2
    kernel.astype(np.float32)
#    a = kernel.sum()
    y_data = signal.fftconvolve(y_data, kernel, mode='same')

    ymax = y_data.max()
    print 'kernel sum:', kernel.sum(), 'ymax:', ymax
    #d = ymax/10
    print y_data
    d = 0.5
    mask_not = y_data < d
    mask_not[:,0:20] = True
    mask_not[:,-20:] = True
    mask_not[0:20,:] = True
    mask_not[-20:,:] = True
    y_data[mask_not] = d
    x,y = max_laser(y_data)

#    y_data = (l2-l1)


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
