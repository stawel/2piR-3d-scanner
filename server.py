#!/usr/bin/python
import io
import socket
import struct
from PIL import Image
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')


cv2.namedWindow('image', cv2.WINDOW_NORMAL)

i = 10000

path = './scans/s/'
extension = '.tif'
if not os.path.exists(path):
    os.mkdir(path)

def get_img_raw():
    image_type = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
    # Read the length of the image as a 32-bit unsigned int. If the
    # length is zero, quit the loop
    image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
    if not image_len:
        image_stream = None
    else:
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_stream.seek(0)

    return image_type, image_len, image_stream

def image_bayer(stream):
    ver = 1
    offset = {
        1: 6404096,
        2: 10270208,
        }[ver]
    data = stream.getvalue()[-offset:]
    assert data[:4] == 'BRCM'
    data = data[32768:]
    data = np.fromstring(data, dtype=np.uint8)

    reshape, crop = {
        1: ((1952, 3264), (1944, 3240)),
        2: ((2480, 4128), (2464, 4100)),
        }[ver]
    data = data.reshape(reshape)[:crop[0], :crop[1]]

    data = data.astype(np.uint16) << 2
#    print 'bayer1:',data[1000,1000:1020]
    for byte in range(4):
        data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
    data = np.delete(data, np.s_[4::5], 1)

#    print 'bayer2:',data[1000,1000:1020]
    rgb = np.zeros(data.shape + (3,), dtype=data.dtype)
    rgb[1::2, 0::2, 0] = data[0::2, 1::2] # Blue
    rgb[0::2, 0::2, 1] = data[0::2, 0::2] # Green
    rgb[1::2, 1::2, 1] = data[1::2, 1::2] # Green
    rgb[0::2, 1::2, 2] = data[1::2, 0::2] # Red

    bayer = np.zeros(rgb.shape, dtype=np.uint8)
    bayer[1::2, 0::2, 0] = 1 # Red
    bayer[0::2, 0::2, 1] = 1 # Green
    bayer[1::2, 1::2, 1] = 1 # Green
    bayer[0::2, 1::2, 2] = 1 # Blue

    output = np.empty(rgb.shape, dtype=rgb.dtype)
    window = (3, 3)
    borders = (window[0] - 1, window[1] - 1)
    border = (borders[0] // 2, borders[1] // 2)

    rgb_pad = np.zeros((
        rgb.shape[0] + borders[0],
        rgb.shape[1] + borders[1],
        rgb.shape[2]), dtype=rgb.dtype)
    rgb_pad[
        border[0]:rgb_pad.shape[0] - border[0],
        border[1]:rgb_pad.shape[1] - border[1],
        :] = rgb
    rgb = rgb_pad

    bayer_pad = np.zeros((
        bayer.shape[0] + borders[0],
        bayer.shape[1] + borders[1],
        bayer.shape[2]), dtype=bayer.dtype)
    bayer_pad[
        border[0]:bayer_pad.shape[0] - border[0],
        border[1]:bayer_pad.shape[1] - border[1],
        :] = bayer
    bayer = bayer_pad

    for plane in range(3):
        p = rgb[..., plane]
        b = bayer[..., plane]
        pview = as_strided(p, shape=(
            p.shape[0] - borders[0],
            p.shape[1] - borders[1]) + window, strides=p.strides * 2)
        bview = as_strided(b, shape=(
            b.shape[0] - borders[0],
            b.shape[1] - borders[1]) + window, strides=b.strides * 2)
        psum = np.einsum('ijkl->ij', pview)
        bsum = np.einsum('ijkl->ij', bview)
        output[..., plane] = psum // bsum
    return output<<6

def image_bayer_small(stream):
    ver = 1
    offset = {
        1: 6404096,
        2: 10270208,
        }[ver]
    data = stream.getvalue()[-offset:]
    assert data[:4] == 'BRCM'
    data = data[32768:]
    data = np.fromstring(data, dtype=np.uint8)

    reshape, crop = {
        1: ((1952, 3264), (1944, 3240)),
        2: ((2480, 4128), (2464, 4100)),
        }[ver]
    data = data.reshape(reshape)[:crop[0], :crop[1]]

    data = data.astype(np.uint16) << 2
#    print 'bayer1:',data[1000,1000:1020]
    for byte in range(4):
        data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11)
    data = np.delete(data, np.s_[4::5], 1)

#    print 'bayer2:',data[1000,1000:1020]
    rgb = np.zeros(data.shape + (3,), dtype=data.dtype)
    rgb[1::2, 0::2, 0] = data[0::2, 1::2] # Blue
    rgb[0::2, 0::2, 1] = data[0::2, 0::2] # Green
    rgb[1::2, 1::2, 1] = data[1::2, 1::2] # Green
    rgb[0::2, 1::2, 2] = data[1::2, 0::2] # Red

    bayer = np.zeros(rgb.shape, dtype=np.uint8)
    bayer[1::2, 0::2, 0] = 1 # Red
    bayer[0::2, 0::2, 1] = 1 # Green
    bayer[1::2, 1::2, 1] = 1 # Green
    bayer[0::2, 1::2, 2] = 1 # Blue

    output = np.empty(rgb.shape, dtype=rgb.dtype)
    window = (3, 3)
    borders = (window[0] - 1, window[1] - 1)
    border = (borders[0] // 2, borders[1] // 2)

    rgb_pad = np.zeros((
        rgb.shape[0] + borders[0],
        rgb.shape[1] + borders[1],
        rgb.shape[2]), dtype=rgb.dtype)
    rgb_pad[
        border[0]:rgb_pad.shape[0] - border[0],
        border[1]:rgb_pad.shape[1] - border[1],
        :] = rgb
    rgb = rgb_pad

    bayer_pad = np.zeros((
        bayer.shape[0] + borders[0],
        bayer.shape[1] + borders[1],
        bayer.shape[2]), dtype=bayer.dtype)
    bayer_pad[
        border[0]:bayer_pad.shape[0] - border[0],
        border[1]:bayer_pad.shape[1] - border[1],
        :] = bayer
    bayer = bayer_pad

    for plane in range(3):
        p = rgb[..., plane]
        b = bayer[..., plane]
        pview = as_strided(p, shape=(
            p.shape[0] - borders[0],
            p.shape[1] - borders[1]) + window, strides=p.strides * 2)
        bview = as_strided(b, shape=(
            b.shape[0] - borders[0],
            b.shape[1] - borders[1]) + window, strides=b.strides * 2)
        psum = np.einsum('ijkl->ij', pview)
        bsum = np.einsum('ijkl->ij', bview)
        output[..., plane] = psum // bsum
    return output<<6

def get_img():
    image_type, image_len, image_stream =  get_img_raw()
    if not image_len:
        return image_len, None

    if image_type == 1:
        return image_len, image_bayer(image_stream)
    else:
        data = np.fromstring(image_stream.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data,cv2.IMREAD_COLOR)
        return image_len, image


try:
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        image_len, image = get_img()
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)
        cv2.imshow('image', image)
        cv2.imwrite(path + str(i) + extension,image)
        i = i + 1

        height, width, channels = image.shape
        print image_len, height, width, channels

finally:
    connection.close()
    server_socket.close()
    cv2.destroyAllWindows()
