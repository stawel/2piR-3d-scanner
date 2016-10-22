#!/usr/bin/python

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2

file_nr = 10032
name1 = str(file_nr) + '.jpg'
name2 = str(file_nr+1) + '.jpg'


N = 30


# Load an color image in grayscale
img1 = cv2.imread(name1,cv2.IMREAD_COLOR)
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV).astype(np.float32)/[180.,255.,255.]

img2 = cv2.imread(name2,cv2.IMREAD_COLOR)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV).astype(np.float32)/[180.,255.,255.]


fig = plt.figure()

ax = fig.add_subplot(1,2,1)
imgplot = plt.imshow(hsv_to_rgb(img2_hsv))
scatter = plt.scatter(50, 50, marker='s', s = 2*N, alpha = 0.1)
(y,x,z) = img2_hsv.shape
plt.axis([0., x, y, 0.])

ax3d = fig.add_subplot(1,2,2, projection='3d')

msize = 1
hsv_r, = ax3d.plot([], [], 'r.', markersize=msize)
hsv_b, = ax3d.plot([], [], 'b.', markersize=msize)
ax3d.set_xlabel('H')
ax3d.set_ylabel('S')
ax3d.set_zlabel('V')


def onclick(event):
    if ax == event.inaxes:
        x = int(event.xdata)
        y = int(event.ydata)
        n_hsv1 = img1_hsv[y-N:y+N,x-N:x+N].reshape(4*N*N,3)
        n_hsv2 = img2_hsv[y-N:y+N,x-N:x+N].reshape(4*N*N,3)
        hsv_b.set_data(n_hsv1[:,0], n_hsv1[:,1])
        hsv_b.set_3d_properties(n_hsv1[:,2])
        hsv_r.set_data(n_hsv2[:,0], n_hsv2[:,1])
        hsv_r.set_3d_properties(n_hsv2[:,2])
        scatter.set_offsets(np.array([x,y]))
#        scatter.y = y

        fig.canvas.draw()
    
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)


cid = fig.canvas.mpl_connect('button_press_event', onclick)


plt.show()
