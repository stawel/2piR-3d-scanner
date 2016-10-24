#!/usr/bin/python

import numpy as np
import cv2
from pi2R.lines import *
from pi2R.point_cloud import *
import math
import cPickle
from timeit import default_timer as timer


path = "./s4/"

filename = 'points2d.dat'

retu = []


for i in range(10000,13100,2):
    start = timer()
    line = Line(path + str(i) + ".jpg", path + str(i+1) + ".jpg")
    rp = line.get_points_2d()
    colors = line.get_colors_rgb()
    print i, len(rp), len(colors), 'time:', timer() - start
    retu.append([i,np.asarray(rp, dtype=np.float32), np.asarray(colors, dtype=np.uint8)])


with open(filename, 'wb') as output:
    cPickle.dump(retu, output, cPickle.HIGHEST_PROTOCOL)
