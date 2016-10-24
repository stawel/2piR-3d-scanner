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

file_nr = 13090
file_nr = 11050

name1 = str(file_nr) + '.jpg'
name2 = str(file_nr+1) + '.jpg'


N = 5

from scipy.signal import argrelmax, argrelmin
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from timeit import default_timer as timer

def bright_mask(h1,s1,v1,h2,s2,v2):
    # bright objects
    mask_bright1 = 0.9 < v1
    mask_bright2 = 0.9 < v2

    mask_satur = s2+0.05 < s1
    return np.logical_and(np.logical_and(mask_bright1, mask_bright2),mask_satur)

def max_laser(h1,v1,s1,h2,v2,s2,  not_red_mask, not_mask):
    start = timer()
    y = (v2-v1)**2 + (s2-s1)**2+(h2-h1)**2
    y[np.logical_or(not_red_mask, not_mask)] = -1.
#    y = gaussian_filter1d(y, sigma=2, axis=1)
    y = gaussian_filter(y, sigma=2)
    y[np.logical_or(not_red_mask, not_mask)] = -1.
    retu = argrelmax(y, axis=1)
    end = timer()
    print ('max_laser time:',end - start)

    return retu

def red_mask(h1,s1,v1,h2,s2,v2):
    # bright objects
    d = 0.1
    mask_red1 = h2 < 2*d
    mask_red2 = -d > h2
    return np.logical_or(mask_red1, mask_red2)

def wrap_red(h):
    w = h>0.5
    h[w]-=1.
    return h

def split(img):
    return wrap_red(img[:,:,0]),img[:,:,1],img[:,:,2]

def transform(img1, img2):
    img1 = img1.copy()
    img2 = img2.copy()
    h1,s1,v1 = split(img1)
    h2,s2,v2 = split(img2)

    res = img2.copy()

    mask_not_brighter = v2 < v1+0.07

    mask_not_b = np.logical_not(bright_mask(h1,s1,v1, h2,s2,v2))
    mask_not_r = np.logical_not(red_mask(h1,s1,v1, h2,s2,v2))

    mask_not = np.logical_and(mask_not_b,mask_not_brighter)

    x,y = max_laser(h1,v1,s1,h2, v2, s2, mask_not_r, mask_not)

    mask =np.logical_or(mask_not_r, np.logical_and(mask_not_brighter,mask_not_b))
#    mask =np.logical_or(mask_not_r, mask_not_brighter)

#    res[mask] = [0.,0.,0.]
#    print x,y
    res[x,y] = [0.5,1.,1.]
#    res[x+1,y] = [0.5,1.,1.]
#    res[x-1,y] = [0.5,1.,1.]
#    res[x,y+1] = [0.5,1.,1.]
#    res[x,y-1] = [0.5,1.,1.]

#    res[mask_not_brighter] = [0.,0.,0.]
#    res[mask_not_b] = [0.,0.,0.]
#    res[mask_not_r] = [0.,0.,0.]
    return res, (x,y)

def get_points_2d_(img1, img2):
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV);
    img1_hsv = img1_hsv.astype(np.float32)/[180.,255.,255.]

    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    img2_hsv = img2_hsv.astype(np.float32)/[180.,255.,255.]
    image, (x,y)
    return x,y

# Load an color image in grayscale
img1 = cv2.imread(name1,cv2.IMREAD_COLOR)
img1 = cv2.GaussianBlur(img1,(15,5),0)

img1_hsv_dis = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV);
img1_hsv = img1_hsv_dis.astype(np.float32)/[180.,255.,255.]

img2 = cv2.imread(name2,cv2.IMREAD_COLOR)
img2 = cv2.GaussianBlur(img2,(15,5),0)

img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

img2_hsv_dis = img2_hsv
img2_hsv = img2_hsv.astype(np.float32)/[180.,255.,255.]

img2_hsv_dis, xy = transform(img1_hsv, img2_hsv)
#img2_hsv_dis = img2_hsv_dis.astype(np.float32)/[180.,255.,255.]


fig = plt.figure()

ax = fig.add_subplot(1,2,1)
imgplot = ax.imshow(hsv_to_rgb(img2_hsv_dis))

rectangle = ax.add_patch(Rectangle((0, 0),2*N,2*N,alpha=0.2))

(y,x,z) = img2_hsv.shape
plt.axis([0., x, y, 0.])

ax3d = fig.add_subplot(1,2,2, projection='3d')

msize = 1
hsv_r, = ax3d.plot([], [], 'r.', markersize=msize)
hsv_b, = ax3d.plot([], [], 'b.', markersize=msize)
ax3d.set_xlabel('H')
ax3d.set_ylabel('S')
ax3d.set_zlabel('V')

r_x, r_y = 0, 0
def set_rectangle_xy(x,y, N):
    global r_x, r_y
    x, y = int(x), int(y)
    r_x, r_y = x, y
    n_hsv1 = img1_hsv[y-N:y+N,x-N:x+N].reshape(4*N*N,3)
    n_hsv2 = img2_hsv[y-N:y+N,x-N:x+N].reshape(4*N*N,3)
    hsv_b.set_data(n_hsv1[:,0], n_hsv1[:,1])
    hsv_b.set_3d_properties(n_hsv1[:,2])
    hsv_r.set_data(n_hsv2[:,0], n_hsv2[:,1])
    hsv_r.set_3d_properties(n_hsv2[:,2])
    rectangle.set_width(2*N)
    rectangle.set_height(2*N)
    rectangle.set_xy(np.array([x-N,y-N]))
    fig.canvas.draw()


def onclick(event):
    if ax == event.inaxes and event.button == 1:
        set_rectangle_xy(event.xdata, event.ydata, N)
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)


def onbutton(event):
    global N, r_x, r_y
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

    set_rectangle_xy(r_x,r_y, N)
    print 'key=', event.key, 'event=', event

cid = fig.canvas.mpl_connect('key_press_event', onbutton)
cid = fig.canvas.mpl_connect('button_press_event', onclick)


plt.show()
