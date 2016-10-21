#!/usr/bin/python

import numpy as np
import cv2
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis, dtype= np.float32)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]], dtype= np.float32)


def v(x,y,z):
    return np.asarray([x,y,z], dtype= np.float32)
def v2(x,y):
    return np.asarray([x,y], dtype= np.float32)

class CamLaser:
    def __init__(self, cam_O, cam_C, cam_DX, cam_DY, cam_resolution, laser_O, laser_N):
        self.cam_O = cam_O  # camer position
        self.cam_C = cam_C  # camer view direction
        self.cam_DX = cam_DX
        self.cam_DY = cam_DY
        self.cam_resolution_2 = cam_resolution/2
        self.laser_O = laser_O # laser position
        self.laser_N = laser_N # lasers plane normal vector
        self.compute_laser_D() # lasers plane D parameter (Ax + By + Cz + D = 0 or N*X + D = 0)

    def rotateCam(self, angle):
        m = rotation_matrix(v(0.,0.,1.), angle)
        self.cam_O = np.dot(m, self.cam_O)
        self.cam_C = np.dot(m, self.cam_C)  # camer view direction
        self.cam_DX = np.dot(m, self.cam_DX)
        self.cam_DY = np.dot(m, self.cam_DY)


    def compute_laser_D(self):
        self.laser_D = -np.dot(self.laser_N, self.laser_O)

    def rotate(self, angle):
        m = rotation_matrix(v(0.,0.,1.), angle)
        self.n_cam_O = np.dot(m, self.cam_O)
        self.n_cam_C = np.dot(m, self.cam_C)  # camer view direction
        self.n_cam_DX = np.dot(m, self.cam_DX)
        self.n_cam_DY = np.dot(m, self.cam_DY)
        self.n_laser_O = np.dot(m, self.laser_O) # laser position
        self.n_laser_N = np.dot(m, self.laser_N) # lasers plane normal vector
        self.compute_laser_D()
        self.calc_()

    def calc_(self):
#        print 'D=', self.laser_D
        self.a = -self.laser_D-np.dot(self.n_laser_N, self.n_cam_O)
#        print 'a=', self.a

    def get_point_3d(self, point_2d):
        point_2d -= self.cam_resolution_2
        z = self.n_cam_C + point_2d[0]*self.n_cam_DX+point_2d[1]*self.n_cam_DY
        return self.a/np.dot(self.n_laser_N, z)*z + self.n_cam_O

    def compute_points_3d(self, p2d):
        if len(p2d) > 0:
            z = np.zeros((p2d.shape[0],p2d.shape[1]+1), dtype=np.float32)
            p2d -= self.cam_resolution_2
            p2d_x, p2d_y = p2d[:,0], p2d[:,1]
            z[:,0] = self.n_cam_DX[0]*p2d_x + self.n_cam_DY[0]*p2d_y + self.n_cam_C[0]
            z[:,1] = self.n_cam_DX[1]*p2d_x + self.n_cam_DY[1]*p2d_y + self.n_cam_C[1]
            z[:,2] = self.n_cam_DX[2]*p2d_x + self.n_cam_DY[2]*p2d_y + self.n_cam_C[2]
            a = self.a/(self.n_laser_N*z).sum(1)
            aa = np.zeros((p2d.shape[0],p2d.shape[1]+1), dtype=np.float32)
            aa[:,0] = a
            aa[:,1] = a
            aa[:,2] = a
            b = aa*z + self.n_cam_O
            return b
            print 'N=',self.n_laser_N
            print 'C=', self.n_cam_C
            print 'DX=', self.n_cam_DX
            print 'DY=', self.n_cam_DY

            print 'z=',z
            print 'a=',a
            print 'aa=',aa
            print 'b=',b
            print '!!!!!!!!!!!'
            sdasd+=1
        return []

    def compute_points_3d_(self, points_2d):
        retu = []
        for i in points_2d:
            x = self.get_point_3d(i)
#            print 'x=',x
            retu.append(x)
        return np.asarray(retu)

class Line:
    def __init__(self, normal_image_name, laser_image_name):
        self.n_img = cv2.imread(normal_image_name, cv2.IMREAD_COLOR)
        self.n_img = cv2.GaussianBlur(self.n_img,(5,5),0)

        self.l_img = cv2.imread(laser_image_name, cv2.IMREAD_COLOR)
        self.l_img = cv2.GaussianBlur(self.l_img,(5,5),0)

    def _do_signal_rgb(self):
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

    def _do_signal_r(self):
        """r"""
        b1,g1,r1 = cv2.split(self.n_img)
        b2,g2,r2 = cv2.split(self.l_img)
        self.signal = cv2.absdiff(r2,r1)


    def _do_mask_rgb(self):
        self._do_signal_rgb()
        r, self.mask = cv2.threshold(self.signal, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return r

    def _do_mask_r(self):
        self._do_signal_r()
        r, self.mask = cv2.threshold(self.signal, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return r

    def _do_mask_r_tresh(self, t):
        self._do_signal_r()
        r, self.mask = cv2.threshold(self.signal, t, 255, cv2.THRESH_BINARY)
        return r

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
#                    print cx,cy
                    retu.append(((cx+cx2)/2,cy))

                cy = y
                cx = x
                cx2 = x
            elif x < cx2+5:
                cx2 = x
            else:
#                print '!', cx,cy
                retu.append(((cx+cx2)/2,cy))
                cx2 = x
                cx = x
        return retu

    def get_points_2d_(self):
#        self._do_mask_rgb()
#        a = np.argwhere(self.mask)
#        if len(a) > 25000:
#            print 'error rgb:', len(a)
        if True:
            r = self._do_mask_r()
            a = np.argwhere(self.mask)
            if len(a) > 25000:
                print 'error r:', len(a)
                self._do_mask_r_tresh(r*2)
                a = np.argwhere(self.mask)
                if len(a) > 25000:
                    print 'error r2:', len(a)
                    return []

        p = self.reduce_pix(a)
        if len(p) > 2500:
            print 'error reduce:', len(p)
            return []
        return p

    def get_points_2d(self):
        self.points_2d = self.get_points_2d_()
        return self.points_2d

    def get_points_3d_flat(self, y = 0):
        self.get_points_2d()
        retu = []
        for i in self.points_2d:
            retu.append((i[0],i[1],y))
        return retu

    def get_points_3d(self, cam_laser):
        self.points_2d = self.get_points_2d()
        retu = []
        for i in self.points_2d:
            x = cam_laser.get_point_3d(i)
#            print 'x=',x
            retu.append(x)
        return retu

    def get_colors(self):
        return [self.n_img[x,y] for y,x in self.points_2d]

    def get_colors_rgb(self):
        return [ [r,g,b] for b,g,r in [self.n_img[x,y] for y,x in self.points_2d]]
