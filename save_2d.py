#!/usr/bin/python

import numpy as np
import cv2
from pi2R.lines import *
from pi2R.point_cloud import *
from pi2R.path_io import *

import math
import cPickle
from timeit import default_timer as timer
import threading
import Queue


path = "./scans/p1/"
out_filename = 'a1.dat'
threads = 8

retu = []

path_info = PathInfo(path)

start_time = timer()

work_queue = Queue.Queue()
result_queue = Queue.Queue()

def worker_thread():
    while not work_queue.empty():
        start = timer()
        i = work_queue.get()
        line = Line(path_info, i)
        rp = line.get_points_2d()
        colors = line.get_colors_rgb()*255.
        colors[colors>255] = 255
        print i, len(rp), len(colors), 'time:', timer() - start
        result_queue.put([i,np.asarray(rp, dtype=np.float32), np.asarray(colors, dtype=np.uint8)])
        work_queue.task_done()

for i in path_info.indexes():
    work_queue.put(i)

for i in range(0, threads):
    worker = threading.Thread(target=worker_thread)
    worker.setDaemon(True)
    worker.start()

work_queue.join()

while not result_queue.empty():
    r = result_queue.get()
    print 'got:', r[0]
    retu.append(r)


with open(out_filename, 'wb') as output:
    cPickle.dump(retu, output, cPickle.HIGHEST_PROTOCOL)

print 'total time:', timer() - start_time
