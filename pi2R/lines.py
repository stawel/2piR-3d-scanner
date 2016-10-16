#!/usr/bin/python

import numpy as np
import cv2


class Line:
    def __init__(self, normal_image_name, laser_image_name):
        self.n_img = cv2.imread(normal_image_name, cv2.IMREAD_COLOR)
        self.l_img = cv2.imread(laser_image_name, cv2.IMREAD_COLOR)

    def _do_signal(self):
        """rgb"""
        b1,g1,r1 = cv2.split(self.n_img)
        b2,g2,r2 = cv2.split(self.l_img)
        rgb2 = b2>>2
        rgb2 +=r2>>2
        rgb2 +=g2>>2
        rgb1 = b1>>2
        rgb1 +=r1>>2
        rgb1 +=g1>>2
        self.signal = cv2.absdiff(rgb2,rgb1)


    def _do_mask(self):
        r, self.mask = cv2.threshold(self.signal, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    def reduce_pix(self, a):
        retu = []
        cy = -1
        cx = -1
        cx2 = 0
        for i in a:
            x = i[1]
            y = i[0]
            if cy != y:
                if cy >= 0:
                    print cx,cy
                    retu.append([(cx+cx2)/2,cy])

                cy = y
                cx = x
                cx2 = x
            elif x < cx2+5:
                cx2 = x
            else:
                print '!', cx,cy
                retu.append([(cx+cx2)/2,cy])
                cx2 = x
                cx = x
        return retu

    def get_points_2d(self):
        self._do_signal()
        self._do_mask()
        self.points_2d = self.reduce_pix(self.mask)
        return self.points_2d



