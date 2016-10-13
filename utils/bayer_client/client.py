#!/usr/bin/python
import io
import socket
import struct
import time
import picamera
import RPi.GPIO as GPIO

# Connect a client socket to my_server:8000 (change my_server to the
# hostname of your server)
client_socket = socket.socket()
client_socket.connect(('192.168.2.214', 8001))

gpio_list = [18]
GPIO.setmode(GPIO.BCM)
GPIO.setup(gpio_list, GPIO.OUT)

state = True
# Make a file-like object out of the connection
connection = client_socket.makefile('wb')
try:
    camera = picamera.PiCamera()
    #camera.resolution = (640, 480)
    camera.resolution = (2592,1944)
    # Start a preview and let the camera warm up for 2 seconds
    camera.start_preview()
    time.sleep(2)

    # Note the start time and construct a stream to hold image data
    # temporarily (we could write it directly to connection but in this
    # case we want to find out the size of each capture first to keep
    # our protocol simple)
    start = time.time()
    stream = io.BytesIO()
    for foo in range(1,100):
        camera.capture(stream, format='jpeg', bayer=True)
        # Write the length of the capture to the stream and flush to
        # ensure it actually gets sent
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
	GPIO.output(18, state)
	state = not state
    # Write a length of zero to the stream to signal we're done
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    client_socket.close()
