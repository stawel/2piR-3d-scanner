#!/usr/bin/python
import io
import socket
import struct
from PIL import Image
import cv2
import numpy as np

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')


cv2.namedWindow('image', cv2.WINDOW_NORMAL)

i = 10000

try:
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        data = np.fromstring(image_stream.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data,cv2.IMREAD_COLOR)
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)
        cv2.imshow('image', image)
        cv2.imwrite(str(i) + '.jpg',image)
        i = i + 1

        height, width, channels = image.shape
        print image_len, height, width, channels

finally:
    connection.close()
    server_socket.close()
    cv2.destroyAllWindows()
