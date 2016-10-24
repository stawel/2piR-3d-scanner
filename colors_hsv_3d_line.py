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

path = './s4/'
name1 = path + str(file_nr) + '.jpg'
name2 = path + str(file_nr+1) + '.jpg'


N = 5


def transform(img1, img2):
    res = img2.copy()
    mask, (x,y) = pi2R.lines2d.transform(img1, img2)
    res[mask] = [0.,0.,0.]
#    print x,y
    res[x,y] = [0.5,1.,1.]
#    res[x+1,y] = [0.5,1.,1.]
    res[x-1,y] = [0.5,1.,1.]
    res[x,y+1] = [0.5,1.,1.]
    res[x,y-1] = [0.5,1.,1.]

    return res, (x,y)


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
