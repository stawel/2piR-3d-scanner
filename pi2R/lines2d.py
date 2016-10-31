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

def transform_(img1, img2, t=3):
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


##########################################################
from scipy.signal import butter, lfilter, freqz, filtfilt
import scipy.ndimage as ndimage
import scipy.signal as signal

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')#, analog=False)
    return b, a

b, a = butter_lowpass(3.667, 30, order=8)

def filter(div):
    b, a = butter_lowpass(3.667, 30, order=8)
    ydiv = filtfilt(b, a, div)
    return ydiv

def filter2(div, size = 10):

    kernel = np.ones((size,size), dtype=np.float32)
    ydiv = signal.fftconvolve(div, kernel, mode='same')
    return ydiv/(size*size)

def subpix(x,y, data):
    points = data[y,[x-1,x,x+1]]
    c = points[1]
    m = points[0]
    p = points[2]
#    c = data[y,x]
#    m = data[y,x-1]
#    p = data[y,x+1]
    xx = (m-p)/(2*(p+m-2*c))
#    print xx+x
    return np.ndarray.astype(xx+x, np.float32),np.ndarray.astype(y, np.float32)

def transform(img1, img2, t=3):
    global kernel, y_data
    h1,l1,s1 = split(img1)
    h2,l2,s2 = split(img2)

    y_t_data = l2-l1
    ydiv = filter2(l1, 15)+0.05
    y_t_data /= ydiv
    (y_size,x_size) = y_t_data.shape

    y_t_data = filtfilt(b, a, y_t_data, axis=1)
    y_t_data[:,0:10] = 0.
    y_t_data[:,-10:] = 0.
    y_t_data[0:10,:] = 0.
    y_t_data[-10:,:] = 0.

    x = np.argmax(y_t_data, axis=1)
    y = np.arange(0, y_size)
    y_data = y_t_data

    mask = y_data[y,x] > 0.1
#    print mask

#    np.set_printoptions(threshold=np.nan)
#    print x.shape
#    print y.shape
    x = x[mask]
    y = y[mask]
    x,y = subpix(x,y,y_t_data)
    return [], (y,x)

####################################################3

def get_points_2d_g(img1, img2):
    start = timer()
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)
    image, (x,y) = transform(img1_hsv, img2_hsv)
    end = timer()
    print 'get_points_2d_g time:',end - start
    return x,y
