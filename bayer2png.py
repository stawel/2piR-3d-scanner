#!/usr/bin/python

from timeit import default_timer as timer

t0 = timer()
import numpy as np
import cv2
from pi2R.path_io import *
import os.path
import sys

t1 = timer()
if len(sys.argv) < 2:
    print sys.argv[0]," [-s] in.bayer"
    print " -s  convert to small png"
    sys.exit()

small = False
argv_name = 1
if(sys.argv[1] == '-s'):
    small = True
    argv_name = 2

filename = sys.argv[argv_name]
name, extension = os.path.splitext(filename)
t2 = timer()
img = open_img(filename, extension, small)
t3 = timer()
cv2.imwrite(name + '.png',img)
t4 = timer()

print 'total time:', t4 - t0, t4-t3, t3-t2, t2-t1, t1-t0
