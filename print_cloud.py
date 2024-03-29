#!/usr/bin/python

import numpy as np
import cv2
from pi2R.lines import *
from pi2R.point_cloud import *
import math
import cPickle
from fractions import Fraction


pc = PointCloud()

filename = 'points2d.dat'
filename = 'a6.dat'
filename = 'a10.dat'
filename = 'a1.dat'



res = (997, 1296)

with open(filename, 'rb') as input:
    inp = cPickle.load(input)


def set_points(x = 0.50):
    lN = v(0.269, 0.65, 0.)
    print 'lN=', lN
    lN = np.dot(rotation_matrix(v(0.,0.,1.), math.pi/2.), lN)
    print 'lN=', lN

    cam_laser = CamLaser(cam_O=v(0.,0.065,0.),cam_C=v(0.,2.,0.),
#                     cam_DX=v(0.855/1944.,0.,0.), cam_DY=v(0.,0.,-0.995/2592.),
#                     cam_DX=v(math.tan(41.*math.pi/180.)/res[0]/2.,0.,0.), cam_DY=v(0.,0.,-math.tan(54.*math.pi/180.)/res[1]/2),
                     cam_DX=v(1.33/res[0],0.,0.), cam_DY=v(0.,0.,-2./res[1]),
                     cam_resolution=v2(res[0],res[1]),
                     laser_N=lN, laser_O=v(-0.37,0.,0.))

#    cam_laser.rotateCam(-x)
    skip = 0
    for i, p2d, colors in inp:
        skip+=1
        if skip % 1 != 0:
            continue
        alfa = -(i-10000)/2.*2.*math.pi/(2048.*3.)
        cam_laser.rotate(alfa)
        rp = cam_laser.compute_points_3d(p2d.copy())
        print i, len(rp), len(colors), 'alfa=', alfa
        awb = (Fraction(249, 256), Fraction(229, 128))
        w = 5.
        c = colors.astype(int)*[w,w*float(awb[0]),w*float(awb[1])]
        c[c>255] = 255
        pc.addPoints(rp, c)

#time total: 2.7085518837  pt: 1.60691690445 3.2901763916e-05 0.0562801361084 1.04532194138
#awb_gains: (Fraction(323, 256), Fraction(53, 32))
#exposure_speed: 365524
#brightness: 50
#digital_gain: 1
#contrast: 0
#analog_gain: 1
#sharpness: 0
#current time: 1478211721.39 qsize: 0


def callback(obj, event):
    v = pc.getSliderValue(obj)
    print 'slider=', v
    pc.removeActors()
    set_points(v)
    pc.addActors()

set_points()
#pc.addSlider(callback)
pc.run()
