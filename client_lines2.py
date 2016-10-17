#!/usr/bin/python

import numpy as np
import cv2
from pi2R.lines import *
from pi2R.point_cloud import *
import math

name1 = '10000.jpg'
name2 = '10001.jpg'


pc = PointCloud()

path = "./s2/"

#    def __init__(self, cam_O, cam_C, cam_DX, cam_DY, cam_resolution, laser_O, laser_N):
def v(x,y,z):
    return np.asarray([x,y,z])
def v2(x,y):
    return np.asarray([x,y])

#lN = v(0.37+0.87,2.,0.)
lN = v(2.,2.,0.)
print 'lN=', lN
lN = np.dot(rotation_matrix([0,0,1], math.pi/2.), lN)
print 'lN=', lN

cam_laser = CamLaser(cam_O=v(0.,0.,0.),cam_C=v(0.,1.,0.),
                     cam_DX=v(0.855/1944.,0.,0.), cam_DY=v(0.,0.,0.995/2592.), cam_resolution=v2(1944.,2592.),
                     laser_N=lN, laser_O=v(-0.37,0.,0.))

for i in range(10000,14500,2):

    line = Line(path + str(i) + ".jpg", path + str(i+1) + ".jpg")
    cam_laser.rotate(i/2.*2.*math.pi/(2048.*3.))
    rp = line.get_points_3d(cam_laser)
    colors = line.get_colors()
#    print line.points_2d
#    print colors
    print i, len(rp), len(colors)

    pc.addPoints(rp, colors)

run(pc)
