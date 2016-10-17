#!/usr/bin/python

import numpy as np
import cv2
from pi2R.lines import *
from pi2R.point_cloud import *

name1 = '10000.jpg'
name2 = '10001.jpg'


pc = PointCloud()

path = "./s2/"

for i in range(10000,11000,2*20):

    line = Line(path + str(i) + ".jpg", path + str(i+1) + ".jpg")
    rp = line.get_points_3d(i*-1)
    colors = line.get_colors()
#    print line.points_2d
#    print colors
    print i, len(rp), len(colors)

    pc.addPoints(rp, colors)

run(pc)
