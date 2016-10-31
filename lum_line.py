#!/usr/bin/python

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib
import pi2R.lines2d
file_nr = 13090
#file_nr = 11050
file_nr = 12000

#file_nr = 14400

file_nr = 10018

path = './scans/p2/'
extension = '.png'
name1 = path + str(file_nr) + extension
name2 = path + str(file_nr+1) + extension


N = 5


def transform(disp_img, img1, img2, t=25):
    res = disp_img.copy()
    mask, (x,y) = pi2R.lines2d.transform(img1, img2, t)

#    res = pi2R.lines2d.y_data.copy()

#    res[mask] = [0.,0.,0.]
    color = [0,1.,1.]
#    color = res.max()
    res[x.astype(int),y.astype(int)] = color
#    res[x+1,y] = color
#    res[x-1,y] = color
#    res[x,y+1] = color
#    res[x,y-1] = color

    return res, (x,y)

def norm2(a):
    a /=a.max()
    return a


# Load an color image in grayscale
img1 = cv2.imread(name1,cv2.IMREAD_UNCHANGED)
img1 = img1.astype(np.float32)/float(np.iinfo(img1.dtype).max)
img1_hls = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)/[360.,1.,1.]


img2 = cv2.imread(name2,cv2.IMREAD_UNCHANGED)
img2 = img2.astype(np.float32)/float(np.iinfo(img2.dtype).max)
img2_hls = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)/[360.,1.,1.]

img1_hls[img1_hls[:,:,0]>0.5]-=[1.,0.,0.]
img2_hls[img2_hls[:,:,0]>0.5]-=[1.,0.,0.]


disp_img = img2[:,:,[2,1,0]]
disp_img2, xy = transform(disp_img, img1_hls, img2_hls)


fig = plt.figure()
ax = fig.add_subplot(1,2,1)

#imgplot = ax.imshow(norm2(pi2R.lines2d.y_data))
imgplot = ax.imshow(disp_img2)

rectangle = ax.add_patch(Rectangle((0, 0),2*N,2*N,alpha=0.2))

(y,x,z) = img2.shape
size_x = x
plt.axis([0., x, y, 0.])

fig.add_subplot(1,2,2)

msize = 1
hls_r, = plt.plot([], [], 'r', markersize=msize)
hls_b, = plt.plot([], [], 'b', markersize=msize)
hls_y, = plt.plot([], [], 'y', markersize=msize)
plt.xlabel('X')
plt.ylabel('L')
plt.axis([0., x, 0., 1.1])

r_x, r_y = 0, 0
r_data = []

from scipy.signal import butter, lfilter, freqz, filtfilt
import scipy.ndimage as ndimage
import scipy.signal as signal

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')#, analog=False)
    return b, a

def filter(div):
    b, a = butter_lowpass(3.667, 30, order=8)
    ydiv = filtfilt(b, a, div)
    return ydiv

def filter2(div, size = 15):

    kernel = np.ones(size, dtype=np.float32)
#    print div, kernel
    ydiv = signal.fftconvolve(div, kernel, mode='same')
    print ydiv.dtype
    return ydiv/size

def filter3(div, size = 15):
    kernel = np.ones(size, dtype=np.float32)
#    print div, kernel
    ydiv = signal.fftconvolve(div, kernel, mode='same')
    print ydiv.dtype
    return ydiv/size


def set_rectangle_xy(x,y, N):
    global r_x, r_y, r_data
    x, y = int(x), int(y)
    r_x, r_y = x, y
    n_hls1 = img1_hls[y:y+1,:]
    n_hls2 = img2_hls[y:y+1,:]
    n_hls3  = n_hls2-n_hls1
#    r_data = pi2R.lines2d.y_data[y-N:y+N,x-N:x+N].reshape(4*N*N)
    color = 1
    div = np.zeros(size_x)
    div[:] = n_hls1[:,:,color]
    ydiv = filter2(div, 15)+0.05
    res = np.zeros(size_x)
    res[:]=n_hls3[:,:,color]/ydiv
    res = filter(res)

#    signal.gaussian(kernel_size_y,kt,)
    hls_b.set_data(range(0,size_x), n_hls1[:,:,color])
    hls_r.set_data(range(0,size_x), n_hls2[:,:,color])
    hls_y.set_data(range(0,size_x), res)

    rectangle.set_width(2*N)
    rectangle.set_height(2*N)
    rectangle.set_xy(np.array([x-N,y-N]))
    fig.canvas.draw()


def onclick(event):
    if ax == event.inaxes and event.button == 1:
        set_rectangle_xy(event.xdata, event.ydata, N)
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)

r_t = 1

def onbutton(event):
    global N, r_x, r_y, r_t, disp_img
    dt=0.1
    if event.key == 'x':
        r_y+=1
    if event.key == 'w':
        r_y-=1
    if event.key == 'a':
        r_x-=1
    if event.key == 'd':
        r_x+=1
    if event.key == 'q':
        N-=1
    if event.key == 'e':
        N+=1
    if event.key == 'r':
        r_t-=dt
    if event.key == 't':
        r_t+=dt

    disp_img2, xy = transform(disp_img, img1_hls, img2_hls, r_t)
    imgplot.set_data(norm2(disp_img2))
#    imgplot.set_data(norm2(pi2R.lines2d.y_data))
    fig.canvas.draw()
    print 'r_t:',r_t
    print 'r_data:', r_data.max(), r_data.min()

    set_rectangle_xy(r_x,r_y, N)
    print 'key=', event.key, 'event=', event

cid = fig.canvas.mpl_connect('key_press_event', onbutton)
cid = fig.canvas.mpl_connect('button_press_event', onclick)


plt.show()
