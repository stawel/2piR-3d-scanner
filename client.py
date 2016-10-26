#!/usr/bin/python
import io
import socket
import struct
import time
import picamera
import RPi.GPIO as GPIO
from fractions import Fraction

from pi2R.hardware import *


stepper_init()
laser_init()


# Connect a client socket to my_server:8000 (change my_server to the
# hostname of your server)
client_socket = socket.socket()
client_socket.connect(('192.168.2.214', 8000))


state = True
pos = 0
# Make a file-like object out of the connection
connection = client_socket.makefile('wb')
try:
    camera = picamera.PiCamera(resolution=(2592,1944), framerate=Fraction(5, 1))
    #camera.resolution = (640, 480)
#    camera.resolution = (2592,1944)
    # Start a preview and let the camera warm up for 2 seconds
    laser(1)
    camera.start_preview()

    camera.shutter_speed = 200*1000
#    camera.contrast=100
    camera.brightness=45
    camera.sharpness=-100

#    camera.iso = 800

    time.sleep(10)
    
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g
    
    print 'awb_gains:', camera.awb_gains
    print 'exposure_speed:',camera.exposure_speed
    print 'brightness:',camera.brightness
    print 'digital_gain:',camera.digital_gain
    print 'contrast:',camera.contrast
#    print 'clock_mode:',camera.clock_mode
    print 'analog_gain:', camera.analog_gain
    print 'sharpness:', camera.sharpness

    laser(0)
#    camera.brightness = 25

    # Note the start time and construct a stream to hold image data
    # temporarily (we could write it directly to connection but in this
    # case we want to find out the size of each capture first to keep
    # our protocol simple)
    start = time.time()
    stream = io.BytesIO()
    for foo in camera.capture_continuous(stream, 'jpeg',burst=False):
        # Write the length of the capture to the stream and flush to
        # ensure it actually gets sent

        print 'awb_gains:', camera.awb_gains
        print 'exposure_speed:',camera.exposure_speed
        print 'brightness:',camera.brightness
        print 'digital_gain:',camera.digital_gain
        print 'contrast:',camera.contrast
#    print 'clock_mode:',camera.clock_mode
        print 'analog_gain:', camera.analog_gain


        length = stream.tell()
        connection.write(struct.pack('<L', length))
        connection.flush()
        # Rewind the stream and send the image data over the wire
        stream.seek(0)
        connection.write(stream.read())
        # If we've been capturing for more than 30 seconds, quit
#        if time.time() - start > 300:
#            break
        # Reset the stream for the next capture
        stream.seek(0)
        stream.truncate()
        print time.time(), length
        state = not state
        if state:
            pos+=1
            stepper_goto(pos,1, delay=0.7)
            laser(0)
            time.sleep(0.3)
        else:
            laser(1)

    # Write a length of zero to the stream to signal we're done
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    client_socket.close()
