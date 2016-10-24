#!/usr/bin/python

import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2
from matplotlib.patches import Rectangle


file_nr = 10032
name1 = str(file_nr) + '.jpg'
name2 = str(file_nr+1) + '.jpg'

N = 15


# Load an color image in grayscale
img1 = cv2.imread(name1,cv2.IMREAD_COLOR)
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV).astype(np.float32)/[180.,255.,255.]

img2 = cv2.imread(name2,cv2.IMREAD_COLOR)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV).astype(np.float32)/[180.,255.,255.]


fig = plt.figure()
#fig.add_subplot(1,3,1)
#imgplot = plt.imshow(hsv_to_rgb(img1_hsv))

ax = fig.add_subplot(1,2,1)
imgplot = plt.imshow(hsv_to_rgb(img2_hsv))
rectangle = ax.add_patch(Rectangle((0, 0),2*N,2*N,alpha=0.2))
(y,x,z) = img2_hsv.shape
plt.axis([0., x, y, 0.])

fig.add_subplot(1,2,2)

msize = 1
hv_r, = plt.plot([], [], 'r.', markersize=msize)
hv_b, = plt.plot([], [], 'b.', markersize=msize)
#plt.xlabel("H")
#plt.ylabel("V")
#plt.axis([0., 1., 0., 1.])

#fig.add_subplot(2,3,6)

hs_r, = plt.plot([], [], 'c.', markersize=msize)
hs_b, = plt.plot([], [], 'g.', markersize=msize)
#plt.xlabel("H")
#plt.ylabel("S")
plt.axis([0., 1., 0., 1.])

r_x, r_y = 0, 0
def set_rectangle_xy(x,y, N):
    global r_x, r_y
    x, y = int(x), int(y)
    r_x, r_y = x, y
    n_hsv1 = img1_hsv[y-N:y+N,x-N:x+N].reshape(4*N*N,3)
    n_hsv2 = img2_hsv[y-N:y+N,x-N:x+N].reshape(4*N*N,3)
    hv_b.set_data(n_hsv1[:,0], n_hsv1[:,2])
    hs_b.set_data(n_hsv1[:,1], n_hsv1[:,2])
    hv_r.set_data(n_hsv2[:,0], n_hsv2[:,2])
    hs_r.set_data(n_hsv2[:,1], n_hsv2[:,2])
    rectangle.set_xy(np.array([x-N,y-N]))

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
