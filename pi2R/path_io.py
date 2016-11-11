#!/usr/bin/python


import numpy as np
import cv2
import math
import scipy.ndimage as ndimage
import scipy.signal as signal
from numpy.lib.stride_tricks import as_strided
import os
import io

from scipy.signal import argrelmax, argrelmin
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from timeit import default_timer as timer

def image_bayer(stream):
    ver = 1
    offset = {
        1: 6404096,
        2: 10270208,
        }[ver]
    data = stream.read()[-offset:]
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

    data_stream = stream.read()
    if(data_stream[:4] == 'BRCM'):
        data = data_stream
    else:
        data = data_stream[-offset:]
    assert data[:4] == 'BRCM'
    data = data[32768:]
    data = np.fromstring(data, dtype=np.uint8)

    reshape, crop = {
        1: ((1952, 3264), (1944, 3240)),
        2: ((2480, 4128), (2464, 4100)),
        }[ver]
    data = data.reshape(reshape)[:crop[0], :crop[1]]

    res_y, res_x = crop
    res_x = res_x*4/5
    ndata = np.zeros((res_y, res_x), dtype=np.uint16)
    for byte in range(4):
        ndata[:, byte::4] = data[:,byte::5]

    shift = 5
    ndata <<= shift+2
    for byte in range(4):
        ndata[:, byte::4] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 0b11) << shift

    rgb = np.zeros((res_y/2,res_x/2,3), dtype=ndata.dtype)
    rgb[:, :, 0] = ndata[0::2, 1::2]*2 # Blue
    rgb[:, :, 1] = ndata[0::2, 0::2]   # Green
    rgb[:, :, 1] += ndata[1::2, 1::2]  # Green
    rgb[:, :, 2] = ndata[1::2, 0::2]*2 # Red

    return rgb


def open_img(filename, extension, small = False):
    if extension == '.bayer':
        t = timer()
        f = io.open(filename, "rb")
        if small:
            img = image_bayer_small(f)
        else:
            img = image_bayer(f)
        f.close()
    else:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return img

class PathInfo:
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)
        _, self.extension = os.path.splitext(self.files[0])
        self.indexes_ = []
        for f in self.files:
            i,_ = os.path.splitext(f)
            self.indexes_.append(int(i))
        list.sort(self.indexes_)
        self.minimum = self.indexes_[0]
        self.maximum = self.indexes_[-1]
        self.maximum += self.maximum & 1

    def indexes(self, step = 1):
        return range(self.minimum, self.maximum , 2*step)

    def get_normal_filename(self, index):
        return self.path + str(index) + self.extension

    def get_laser_filename(self, index):
        return self.path + str(index + 1) + self.extension

    def open_img(self, filename):
        img = open_img(filename, self.extension, True)
        (y,x,d) = img.shape
        if x>y:
            img = cv2.transpose(img)
            img = cv2.flip(img, 0)

        img = img.astype(np.float32)/float(np.iinfo(img.dtype).max)
        return img

    def open_normal_img(self, index):
        filename = self.get_normal_filename(index)
        return self.open_img(filename)

    def open_laser_img(self, index):
        filename = self.get_laser_filename(index)
        return self.open_img(filename)
