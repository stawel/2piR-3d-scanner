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
file_nr = 11050

path = './s4/'
name1 = path + str(file_nr) + '.jpg'
name2 = path + str(file_nr+1) + '.jpg'


N = 5


def transform(disp_img, img1, img2, t=25):
    res = disp_img
    mask, (x,y) = pi2R.lines2d.transform(img1, img2, t)

    res[mask] = [0.,0.,0.]
    color = [0,255,255]
    res[x,y] = color
#    res[x+1,y] = color
    res[x-1,y] = color
    res[x,y+1] = color
    res[x,y-1] = color

    return res, (x,y)

def norm2(a):
    a /=a.max()
    return a


# Load an color image in grayscale
img1 = cv2.imread(name1,cv2.IMREAD_COLOR)
#img1 = cv2.GaussianBlur(img1,(15,5),0)
img1_hls = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS);
img1_hls = img1_hls.astype(np.float32)/[180.,255.,255.]

img2 = cv2.imread(name2,cv2.IMREAD_COLOR)
#img2 = cv2.GaussianBlur(img2,(15,5),0)
img2_hls = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)
img2_hls = img2_hls.astype(np.float32)/[180.,255.,255.]

disp_img = img2[:,:,[2,1,0]]

disp_img, xy = transform(disp_img, img1_hls, img2_hls)


fig = plt.figure()

ax = fig.add_subplot(1,2,1)

imgplot = ax.imshow(disp_img)
#imgplot = ax.imshow(norm2(pi2R.lines2d.y_data))

rectangle = ax.add_patch(Rectangle((0, 0),2*N,2*N,alpha=0.2))

(y,x,z) = img2.shape
plt.axis([0., x, y, 0.])

ax3d = fig.add_subplot(1,2,2, projection='3d')

msize = 1
hls_r, = ax3d.plot([], [], 'r.', markersize=msize)
hls_b, = ax3d.plot([], [], 'b.', markersize=msize)
ax3d.set_xlabel('H')
ax3d.set_ylabel('L')
ax3d.set_zlabel('S')

r_x, r_y = 0, 0
def set_rectangle_xy(x,y, N):
    global r_x, r_y
    x, y = int(x), int(y)
    r_x, r_y = x, y
    n_hls1 = img1_hls[y-N:y+N,x-N:x+N].reshape(4*N*N,3)
    n_hls2 = img2_hls[y-N:y+N,x-N:x+N].reshape(4*N*N,3)
    hls_b.set_data(n_hls1[:,0], n_hls1[:,1])
    hls_b.set_3d_properties(n_hls1[:,2])
    hls_r.set_data(n_hls2[:,0], n_hls2[:,1])
    hls_r.set_3d_properties(n_hls2[:,2])
    rectangle.set_width(2*N)
    rectangle.set_height(2*N)
    rectangle.set_xy(np.array([x-N,y-N]))
    fig.canvas.draw()


def onclick(event):
    if ax == event.inaxes and event.button == 1:
        set_rectangle_xy(event.xdata, event.ydata, N)
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        event.button, event.x, event.y, event.xdata, event.ydata)

r_t = 15.

def onbutton(event):
    global N, r_x, r_y, r_t, disp_img
    dt=1
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

    disp_img, xy = transform(disp_img, img1_hls, img2_hls, r_t)
    imgplot.set_data(norm2(pi2R.lines2d.y_data))
    fig.canvas.draw()
    print 'r_t:',r_t


    set_rectangle_xy(r_x,r_y, N)
    print 'key=', event.key, 'event=', event

cid = fig.canvas.mpl_connect('key_press_event', onbutton)
cid = fig.canvas.mpl_connect('button_press_event', onclick)


plt.show()
