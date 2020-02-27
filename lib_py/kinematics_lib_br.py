#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

import cPickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom


import pickle
#import tf.transformations as tft


# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)




class KinematicsLib():

    def batch_rodrigues(self, theta):
        # theta N x 3
        batch_size = theta.shape[0]

        # print theta[0, :], 'THETA'
        l1norm = torch.norm(theta + 1e-8, p=2, dim=2)
        angle = torch.unsqueeze(l1norm, -1)
        # print angle[0, :], 'ANGLE'
        normalized = torch.div(theta, angle)
        # print normalized[0, :], 'NORM'
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = torch.cat([v_cos, v_sin * normalized], dim=2)
        # print quat[0, :]

        """Convert quaternion coefficients to rotation matrix.
        Args:
            quat: size = [B, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        norm_quat = quat
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=2, keepdim=True)

        # print norm_quat.shape

        w, x, y, z = norm_quat[:, :, 0], norm_quat[:, :, 1], norm_quat[:, :, 2], norm_quat[:, :, 3]
        #
        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                              2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                              2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=2)

        # print "got R"
        return rotMat

    def batch_euler_to_R(self, theta, zeros_cartesian, ones_cartesian):
        batch_size_current = theta.size()[0]

        cosx = torch.cos(theta[:, :, 0])
        sinx = torch.sin(theta[:, :, 0])
        cosy = torch.cos(theta[:, :, 1])
        siny = torch.sin(theta[:, :, 1])
        cosz = torch.cos(theta[:, :, 2])
        sinz = torch.sin(theta[:, :, 2])

        b_zeros = zeros_cartesian[:batch_size_current, :]
        b_ones = ones_cartesian[:batch_size_current, :]

        R_x = torch.stack([b_ones, b_zeros, b_zeros,
                           b_zeros, cosx, -sinx,
                           b_zeros, sinx, cosx], dim=2) \
            .view(batch_size_current, 24, 3, 3)

        R_y = torch.stack([cosy, b_zeros, siny,
                           b_zeros, b_ones, b_zeros,
                           -siny, b_zeros, cosy], dim=2) \
            .view(batch_size_current, 24, 3, 3)

        R_z = torch.stack([cosz, -sinz, b_zeros,
                           sinz, cosz, b_zeros,
                           b_zeros, b_zeros, b_ones], dim=2) \
            .view(batch_size_current, 24, 3, 3)

        R_x = R_x.view(batch_size_current * 24, 3, 3)
        R_y = R_y.view(batch_size_current * 24, 3, 3)
        R_z = R_z.view(batch_size_current * 24, 3, 3)

        R = torch.bmm(torch.bmm(R_z, R_y), R_x).view(batch_size_current, 24, 3, 3)
        return R

    def batch_global_rigid_transformation(self, Rs, Js, parent, GPU, rotate_base=False):
        N = Rs.shape[0]
        if rotate_base:
            np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
            np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
            if GPU == True:
                rot_x = Variable(torch.from_numpy(np_rot_x).float()).cuda()
            else:
                rot_x = Variable(torch.from_numpy(np_rot_x).float())
            root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]
        Js = torch.unsqueeze(Js, -1)

        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            if GPU == True:
                t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).cuda()], dim=1)
            else:
                t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1))], dim=1)
            return torch.cat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]

        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)

        results = torch.stack(results, dim=1)

        new_J = results[:, :, :3, 3]
        if GPU == True:
            Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1)).cuda()], dim=2)
        else:
            Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1))], dim=2)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        A = results - init_bone

        return new_J, A


    def batch_dir_cos_angles_from_euler_angles(self, theta, zeros_cartesian, ones_cartesian):
        batch_size_current = theta.size()[0]

        cosx = torch.cos(theta[:, :, 0])
        sinx = torch.sin(theta[:, :, 0])
        cosy = torch.cos(theta[:, :, 1])
        siny = torch.sin(theta[:, :, 1])
        cosz = torch.cos(theta[:, :, 2])
        sinz = torch.sin(theta[:, :, 2])

        b_zeros = zeros_cartesian[:batch_size_current, :]
        b_ones = ones_cartesian[:batch_size_current, :]

        R_x = torch.stack([b_ones, b_zeros, b_zeros,
                           b_zeros, cosx, -sinx,
                           b_zeros, sinx, cosx], dim=2) \
            .view(batch_size_current, 24, 3, 3)

        R_y = torch.stack([cosy, b_zeros, siny,
                           b_zeros, b_ones, b_zeros,
                           -siny, b_zeros, cosy], dim=2) \
            .view(batch_size_current, 24, 3, 3)

        R_z = torch.stack([cosz, -sinz, b_zeros,
                           sinz, cosz, b_zeros,
                           b_zeros, b_zeros, b_ones], dim=2) \
            .view(batch_size_current, 24, 3, 3)

        R_x = R_x.view(batch_size_current * 24, 3, 3)
        R_y = R_y.view(batch_size_current * 24, 3, 3)
        R_z = R_z.view(batch_size_current * 24, 3, 3)

        R = torch.bmm(torch.bmm(R_z, R_y), R_x).view(batch_size_current, 24, 3, 3)

        m00 = R[:, :, 0, 0]
        m01 = R[:, :, 0, 1]
        m02 = R[:, :, 0, 2]
        m10 = R[:, :, 1, 0]
        m11 = R[:, :, 1, 1]
        m12 = R[:, :, 1, 2]
        m20 = R[:, :, 2, 0]
        m21 = R[:, :, 2, 1]
        m22 = R[:, :, 2, 2]

        print m00.size()

        print b_zeros.size()

        # symmetric matrix K
        K = torch.stack([m00 - m11 - m22, 0.0*m00, 0.0*m00, 0.0*m00,
                      m01 + m10, m11 - m00 - m22, 0.0*m00, 0.0*m00,
                      m02 + m20, m12 + m21, m22 - m00 - m11, 0.0*m00,
                      m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22])
        K = K.permute(1, 2, 0).view(-1, 24, 4, 4)
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue

        K = K.detach().cpu().numpy()
        w, V = np.linalg.eigh(K)

        quat = V[:, :, [3, 0, 1, 2], :]
        quat = quat[:, np.arange(24), :, np.argmax(w, axis=2)[0]]
        quat = np.swapaxes(quat, 0, 1)

        neg_multiplier = np.copy(quat[:, :, 0])
        neg_multiplier[neg_multiplier < 0] = -1.
        neg_multiplier[neg_multiplier >= 0] = 1.

        neg_multiplier = np.swapaxes(np.swapaxes(np.stack(4*[neg_multiplier]), 0, 1), 1, 2)

        quat = neg_multiplier*quat - 0.000001

        phi = 2 * np.arccos(quat[:, :, 0])
        dir_cos_angles = np.zeros_like(quat)
        dir_cos_angles = dir_cos_angles[:, :, 0:3]
        dir_cos_angles[:, :, 0] = quat[:, :, 1] * phi / np.sin(phi / 2)
        dir_cos_angles[:, :, 1] = quat[:, :, 2] * phi / np.sin(phi / 2)
        dir_cos_angles[:, :, 2] = quat[:, :, 3] * phi / np.sin(phi / 2)
        return dir_cos_angles



    def batch_euler_angles_from_dir_cos_angles(self, theta_all):

        angle = torch.norm(theta_all + 1e-8, p=2, dim=2)

        num_joints = theta_all.size()[1]

        angle_repeated = angle.repeat(1, 3).view(-1, 3, num_joints).permute(0, 2, 1)
        normalized = torch.div(theta_all, angle_repeated)
        angle = angle * 0.5

        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)

        Q = torch.stack([v_cos, v_sin * normalized[:, :, 0], v_sin * normalized[:, :, 1], v_sin * normalized[:, :, 2]]).permute(1, 2, 0)


        R = torch.stack([1 - 2 * (Q[:,:,2] * Q[:,:,2] + Q[:,:,3] * Q[:,:,3]), 2 * (Q[:,:,1] * Q[:,:,2] - Q[:,:,0] * Q[:,:,3]), 2 * (Q[:,:,0] * Q[:,:,2] + Q[:,:,1] * Q[:,:,3]),
                      2 * (Q[:,:,1] * Q[:,:,2] + Q[:,:,0] * Q[:,:,3]), 1 - 2 * (Q[:,:,1] * Q[:,:,1] + Q[:,:,3] * Q[:,:,3]), 2 * (Q[:,:,2] * Q[:,:,3] - Q[:,:,0] * Q[:,:,1]),
                      2 * (Q[:,:,1] * Q[:,:,3] - Q[:,:,0] * Q[:,:,2]), 2 * (Q[:,:,0] * Q[:,:,1] + Q[:,:,2] * Q[:,:,3]), 1 - 2 * (Q[:,:,1] * Q[:,:,1] + Q[:,:,2] * Q[:,:,2])])

        R = R.permute(1, 2, 0).view(-1, num_joints, 3, 3)

        sy = torch.sqrt(R[:, :, 0, 0] * R[:, :, 0, 0] + R[:, :, 1, 0] * R[:, :, 1, 0])
        #print sy[0, 2]

        singular = sy < 1e-6

        #this will FAIL in the case of a singularity. should be checked according to better way. use w caution
        x = torch.atan2(R[:, :, 2, 1], R[:, :, 2, 2])
        y = torch.atan2(-R[:, :, 2, 0], sy)
        z = torch.atan2(R[:, :, 1, 0], R[:, :, 0, 0])

        euler_angles = torch.stack([x, y, z]).permute(1, 2, 0)
        return euler_angles

