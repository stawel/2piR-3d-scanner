#!/usr/bin/python

import numpy as np
import cv2
from pi2R.lines import *
from pi2R.point_cloud import *
import math
import cPickle
from timeit import default_timer as timer
import threading
import Queue


path = "./scans/p2/"
extension = '.png'
out_filename = 'a3.dat'
threads = 8

retu = []

work_queue = Queue.Queue()
result_queue = Queue.Queue()

def worker_thread():
    while not work_queue.empty():
        start = timer()
        i = work_queue.get()
        line = Line(path + str(i) + extension, path + str(i+1) + extension)
        rp = line.get_points_2d()
        colors = line.get_colors_rgb()*256.
        print i, len(rp), len(colors), 'time:', timer() - start
        result_queue.put([i,np.asarray(rp, dtype=np.float32), np.asarray(colors, dtype=np.uint8)])
        work_queue.task_done()


for i in range(10000,15860,2):
    work_queue.put(i)

for i in range(1, threads):
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
