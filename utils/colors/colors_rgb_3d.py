#!/usr/bin/python

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
import cv2


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

ax = fig.add_subplot(1,2,1)
imgplot = plt.imshow(hsv_to_rgb(img2_hsv))
rectangle = ax.add_patch(Rectangle((0, 0),2*N,2*N,alpha=0.2))
(y,x,z) = img2_hsv.shape
plt.axis([0., x, y, 0.])

ax3d = fig.add_subplot(1,2,2, projection='3d')

msize = 1
rgb_r, = ax3d.plot([], [], 'r.', markersize=msize)
rgb_b, = ax3d.plot([], [], 'b.', markersize=msize)
ax3d.set_xlabel('R')
ax3d.set_ylabel('G')
ax3d.set_zlabel('B')


def onclick(event):
    if ax == event.inaxes and event.button == 1:
        x = int(event.xdata)
        y = int(event.ydata)
        n_bgr1 = img1[y-N:y+N,x-N:x+N].reshape(4*N*N,3)/255.
        n_bgr2 = img2[y-N:y+N,x-N:x+N].reshape(4*N*N,3)/255.
        rgb_b.set_data(n_bgr1[:,2], n_bgr1[:,1])
        rgb_b.set_3d_properties(n_bgr1[:,0])
        rgb_r.set_data(n_bgr2[:,2], n_bgr2[:,1])
        rgb_r.set_3d_properties(n_bgr2[:,0])
        rectangle.set_xy(np.array([x-N,y-N]))

        fig.canvas.draw()
    
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)


cid = fig.canvas.mpl_connect('button_press_event', onclick)


plt.show()
