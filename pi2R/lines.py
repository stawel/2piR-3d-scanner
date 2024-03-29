#!/usr/bin/python

import numpy as np
import cv2
import math
import lines2d

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
        self.a = -self.laser_D-np.dot(self.n_laser_N, self.n_cam_O)

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
    def __init__(self, path_info, index):
        self.n_img_name = path_info.get_normal_filename(index)
        self.n_img = path_info.open_normal_img(index)
        self.l_img = path_info.open_laser_img(index)

    def get_points_2d_(self):
        (x,y) = lines2d.get_points_2d_g(self.n_img, self.l_img)
        if len(x) > 3500:
            print 'error reduce:', len(x), 'name:', self.n_img_name
            return []
        return np.asarray(zip(y,x))

    def get_points_2d(self):
        self.points_2d = self.get_points_2d_()
        self.points_2d_int = self.points_2d.astype(int)

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
        return np.array([self.n_img[x,y] for y,x in self.points_2d_int])

    def get_colors_rgb(self):
        return np.array([ [r,g,b] for b,g,r in [self.n_img[x,y] for y,x in self.points_2d_int]])
