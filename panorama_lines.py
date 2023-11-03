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
import pi2R.path_io
import pi2R.panorama2d


path_info  = pi2R.path_io.PathInfo('./scans/p1/')

panorama = pi2R.panorama2d.Panorama2D(path_info)
panorama.find_angles()


N = 5
file_nr = 11000


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



def open_img(file_nr):
    global disp_img, disp_img2, img1, img1_hls, img2, img2_hls

    img1 = path_info.open_normal_img(file_nr)
    img1_hls = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)/[360.,1.,1.]

    img2 = path_info.open_laser_img(file_nr)
    img2_hls = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)/[360.,1.,1.]

    img1_hls[img1_hls[:,:,0]>0.5]-=[1.,0.,0.]
    img2_hls[img2_hls[:,:,0]>0.5]-=[1.,0.,0.]

    disp_img = img2[:,:,[2,1,0]]
    disp_img2, xy = transform(disp_img, img1_hls, img2_hls)


fig = plt.figure()
ax = fig.add_subplot(1,2,1)
open_img(file_nr)

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

def set_rectangle_xy(x,y, N):
    global r_x, r_y
    x, y = int(x), int(y)
    r_x, r_y = x, y
    n_hls1 = img1_hls[y:y+1,:]
    n_hls2 = img2_hls[y:y+1,:]
    n_hls3  = n_hls2-n_hls1
    color = 1

    hls_b.set_data(range(0,size_x), n_hls1[:,:,color])
    hls_r.set_data(range(0,size_x), n_hls2[:,:,color])
    hls_y.set_data(range(0,size_x), pi2R.lines2d.y_data[y:y+1,:])

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
    global N, r_x, r_y, r_t, disp_img, file_nr, disp_img2
    dt=0.1
    d_file_nr = 0
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

    if event.key == '7':
        d_file_nr+=1
    if event.key == '8':
        d_file_nr+=10
    if event.key == '9':
        d_file_nr+=100
    if event.key == '1':
        d_file_nr-=1
    if event.key == '2':
        d_file_nr-=10
    if event.key == '3':
        d_file_nr-=100

    if event.key == '5':
        imgplot.set_data(disp_img)
    elif d_file_nr != 0:
        file_nr+=d_file_nr*2
        print 'img: ', file_nr
        open_img(file_nr)
        imgplot.set_data(disp_img2)
    else:
        disp_img2, xy = transform(disp_img, img1_hls, img2_hls, r_t)
        imgplot.set_data(norm2(disp_img2))
#    imgplot.set_data(norm2(pi2R.lines2d.y_data))
    fig.canvas.draw()
    print 'r_t:',r_t

    set_rectangle_xy(r_x,r_y, N)
    print 'key=', event.key, 'event=', event

cid = fig.canvas.mpl_connect('key_press_event', onbutton)
cid = fig.canvas.mpl_connect('button_press_event', onclick)


plt.show()
