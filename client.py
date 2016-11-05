#!/usr/bin/python
import io
import socket
import struct
import time
import picamera
import RPi.GPIO as GPIO
import cv2
import numpy as np
import zlib
from numpy.lib.stride_tricks import as_strided

from fractions import Fraction
from pi2R.hardware import *
import threading
import Queue
import os



img_type = 2
img_type = 1 #bayer
write_to_file = True
path = './s/'


sensor_version = 1

stepper_init()
laser_init()

# Connect a client socket to my_server:8000 (change my_server to the
# hostname of your server)
client_socket = socket.socket()
client_socket.connect(('192.168.2.214', 8000))
#client_socket.setblocking(0)
# Make a file-like object out of the connection
connection = client_socket.makefile('wb')#, 20*1000*1000)

network_queue = Queue.Queue()

def network_thread():
    file_nr = 10000
    skip = 1
    if write_to_file:
        skip = 5
        if not os.path.exists(path):
            os.mkdir(path)

    while True:
        r = network_queue.get()
        length = len(r)
        if write_to_file:
            f = open(path+str(file_nr) + '.bayer', 'w')
            f.write(r)
            f.close()
        if file_nr % skip == 0:
            connection.write(struct.pack('<L', img_type))
            connection.write(struct.pack('<L', length))
            connection.write(r)
        file_nr+=1
        network_queue.task_done()

worker = threading.Thread(target=network_thread)
worker.setDaemon(True)
worker.start()

state = True
pos = 0

def camera_info(camera):
    print 'awb_gains:', camera.awb_gains
    print 'exposure_speed:',camera.exposure_speed
    print 'brightness:',camera.brightness
    print 'digital_gain:',camera.digital_gain
    print 'contrast:',camera.contrast
#    print 'clock_mode:',camera.clock_mode
    print 'analog_gain:', camera.analog_gain
    print 'sharpness:', camera.sharpness

def get_bayer_offset():
    offset = {
        1: 6404096,
        2: 10270208,
        }[sensor_version]
    return offset

def image_bayer_small(stream):
    offset = get_bayer_offset()
    data = stream.getvalue()[-offset:]
    assert data[:4] == 'BRCM'
    data = data[32768:]
    data = np.fromstring(data, dtype=np.uint8)

    reshape, crop = {
        1: ((1952, 3264), (1944, 3240)),
        2: ((2480, 4128), (2464, 4100)),
        }[sensor_version]
    data = data.reshape(reshape)[:crop[0], :crop[1]]

    data = data.astype(np.uint16) << 2
#    print 'bayer1:',data[1000,1000:1020]
    for byte in range(4):
        data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
    data = np.delete(data, np.s_[4::5], 1)

#    print 'bayer2:',data[1000,1000:1020]
    (y,x) = data.shape
    rgb = np.zeros((y/2,x/2,3), dtype=data.dtype)
    rgb[:, :, 0] = data[0::2, 1::2]*2 # Blue
    rgb[:, :, 1] = data[0::2, 0::2]   # Green
    rgb[:, :, 1] += data[1::2, 1::2]  # Green
    rgb[:, :, 2] = data[1::2, 0::2]*2 # Red

    return rgb<<5

def make_img(data_stream, stream):
    t1 = time.time()
    img = image_bayer_small(data_stream)
    t2 = time.time()
    r, buf = cv2.imencode('.png', img)
    t3 = time.time()
    data_stream.seek(0)
    data_stream.truncate()
    stream.write(bytearray(buf))
    t4 = time.time()
    print 'make img time:', t2-t1,t3-t2,t4-t3
    return stream

def make_img2(data_stream, stream):
    return data_stream


t =1;
if img_type == 1:
    t = 20

try:
    camera = picamera.PiCamera(resolution=(2592/t,1944/t), framerate=Fraction(5, 2))
    #camera.resolution = (640, 480)
#    camera.resolution = (2592,1944)
    # Start a preview and let the camera warm up for 2 seconds
    laser(1)
    camera.start_preview()

    camera.shutter_speed = 2*200*1000
#    camera.contrast=100
#    camera.brightness=45
#    camera.sharpness=-100

#    camera.iso = 800

    time.sleep(5)
    
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g
    

    camera_info(camera)
    laser(0)
#    camera.brightness = 25

    # Note the start time and construct a stream to hold image data
    # temporarily (we could write it directly to connection but in this
    # case we want to find out the size of each capture first to keep
    # our protocol simple)
    start = time.time()
    stream = io.BytesIO()
    data_stream = io.BytesIO()

    if img_type == 1:
        img_bayer = True
    else:
        img_bayer = False

    for foo in range(1,20000):
        t1 = time.time()
        camera.capture(data_stream, 'jpeg', bayer=img_bayer)
        # Write the length of the capture to the stream and flush to
        # ensure it actually gets sent
        t2 = time.time()
        stream = make_img2(data_stream, stream)
        t3 = time.time()

        camera_info(camera)

        length = stream.tell()
        if img_type == 1:
            offset = length - get_bayer_offset()
        else:
            offset = 0

#        connection.flush()
        # Rewind the stream and send the image data over the wire
        stream.seek(offset)
        r = stream.read()
        network_queue.put(r)
        t4 = time.time()

        # If we've been capturing for more than 30 seconds, quit
#        if time.time() - start > 300:
#            break
        # Reset the stream for the next capture
        stream.seek(0)
        stream.truncate()
        print 'current time:', time.time(), 'qsize:', network_queue.qsize()
        state = not state
        if state:
            pos+=1
            stepper_goto(pos,1, delay=0.7)
            laser(0)
            if network_queue.qsize() >= 6:
                while network_queue.qsize() != 0:
                    print 'waiting queue:', network_queue.qsize()
                    time.sleep(1)
            time.sleep(0.3)
        else:
            laser(1)
        t5 = time.time()
        print 'time total:', t5-t1, ' pt:', t2-t1,t3-t2,t4-t3,t5-t4

    # Write a length of zero to the stream to signal we're done
    connection.write(struct.pack('<L', img_type))
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    client_socket.close()
