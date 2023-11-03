#!/usr/bin/python


import numpy as np
import cv2
import math
import scipy.ndimage as ndimage
import scipy.signal as signal


from scipy.signal import argrelmax, argrelmin
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from timeit import default_timer as timer


from scipy.signal import butter, lfilter, freqz, filtfilt
import scipy.ndimage as ndimage
import scipy.signal as signal
import imutils

def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray*=255
    print imutils.is_cv3()
    gray = gray.astype(np.uint8)
    detector = cv2.FeatureDetector_create("SIFT")
    kps = detector.detect(gray)
    print gray
    extractor = cv2.DescriptorExtractor_create("SIFT")
    (kps, features) = extractor.compute(gray, kps)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)

def find_img_angle(img1, ing2):
    print detect_and_describe(img1)
    print detect_and_describe(img2)

class Panorama2D:
    def __init__(self, path_info):
        self.path_info = path_info
        self.check_distance = 100

    def find_angles(self):
        idx = self.path_info.indexes(self.check_distance)
    #    for i in range(0,len(idx)-1):
        i = 0
        img1 = self.path_info.open_normal_img(idx[i])
        img2 = self.path_info.open_normal_img(idx[i+1])
        find_img_angle(img1,img2)
    #    break
