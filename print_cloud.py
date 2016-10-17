#!/usr/bin/python

import numpy as np
import cv2
from pi2R.lines import *
from pi2R.point_cloud import *
import math
import cPickle

pc = PointCloud()

filename = 'points2d.dat'
filename = 'a5.dat'

#lN = v(0.37+0.87,2.,0.)
lN = v(0.37, 0.65, 0.)
#lN = v(2.,2.,0.)
print 'lN=', lN
lN = np.dot(rotation_matrix([0,0,1], math.pi/2.), lN)
print 'lN=', lN

cam_laser = CamLaser(cam_O=v(0.,0.,0.),cam_C=v(0.,1.,0.),
                     cam_DX=v(0.855/1944.,0.,0.), cam_DY=v(0.,0.,0.995/2592.), cam_resolution=v2(1944.,2592.),
                     laser_N=lN, laser_O=v(-0.37,0.,0.))



with open(filename, 'rb') as input:
    inp = cPickle.load(input)


skip = 0
for i, p2d, colors in inp:
    skip+=1
#    if skip % 5 != 0:
#        continue

    cam_laser.rotate(i/2.*2.*math.pi/(2048.*3.))
    rp = cam_laser.compute_points_3d(p2d)#np.asarray(p2d, dtype=np.float64))
    print i, len(rp), len(colors)

    pc.addPoints(rp, colors)

run(pc)
