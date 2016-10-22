#!/usr/bin/python

import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2

name1 = '10334.jpg'
name2 = '10335.jpg'

N = 30


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
scatter = plt.scatter(50, 50, marker='s', s = N, alpha = 0.1)
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


def onclick(event):
    if ax == event.inaxes:
        x = int(event.xdata)
        y = int(event.ydata)
        n_hsv1 = img1[y:y+N,x:x+N].reshape(N*N,3)/255.
        n_hsv2 = img2[y:y+N,x:x+N].reshape(N*N,3)/255.
        hv_b.set_data(n_hsv1[:,0], n_hsv1[:,2])
        hs_b.set_data(n_hsv1[:,1], n_hsv1[:,2])
        hv_r.set_data(n_hsv2[:,0], n_hsv2[:,2])
        hs_r.set_data(n_hsv2[:,1], n_hsv2[:,2])
        scatter.set_offsets(np.array([x+N/2,y+N/2]))
#        scatter.y = y

        fig.canvas.draw()
    
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)


cid = fig.canvas.mpl_connect('button_press_event', onclick)


plt.show()
