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
#from skimage.feature import hog
#from skimage import data, color, exposure

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import svm, linear_model, decomposition, kernel_ridge, neighbors
from sklearn import metrics
from sklearn.utils import shuffle


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




    def get_bed_distance(self, images, targets, bedangle = None):
        mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)

        try:
            images = images.data.numpy()
            targets = targets.data.numpy()/1000
            try:
                test = targets.shape[2]
            except:
                targets = np.reshape(targets, (targets.shape[0], targets.shape[1]/3, 3))
            bedangle = images[:, -1, 10, 10]


        except:
            images = np.expand_dims(images, axis = 0)
            targets = np.reshape(targets, (10, 3))
            targets = np.expand_dims(targets, axis = 0)
            bedangle = np.expand_dims(bedangle, axis = 0)



        distances = np.zeros((images.shape[0], targets.shape[1]))
        queue_frame = np.zeros((targets.shape[0], targets.shape[1], 4))
        queue_head = np.zeros((targets.shape[0], targets.shape[1], 4))

        # get the shortest distance from the main frame of the bed. it's just the z.
        queue_frame[:, :, 0] = targets[:, :, 2]

        # get the shortest distance from the head of the bed. you have to rotate about the bending point.


        By = (51) * 0.0286 - 0.0286 * 3 * np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis = 1)))
        queue_head[:, :, 0] = targets[:, :, 2] * np.cos(np.deg2rad(np.expand_dims(bedangle[:], axis = 1))) - (targets[:, :,1] - By) * np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis = 1) ))

        # Get the distance off the side of the bed.  The bed is 27 pixels wide, so things over hanging this are added
        queue_frame[:, :, 1] = (-targets[:, :, 0] + 10 * 0.0286).clip(min=0)
        queue_frame[:, :, 1] = queue_frame[:, :, 1] + (targets[:, :, 0] - 37 * 0.0286).clip(min=0)
        queue_head[:, :, 1] = np.copy(queue_frame[:, :, 1])  # the x distance does not depend on bed angle.

        # Now take the Euclidean for each frame and head set
        queue_frame[:, :, 2] = np.sqrt(np.square(queue_frame[:, :, 0]) + np.square(queue_frame[:, :, 1]))
        queue_head[:, :, 2] = np.sqrt(np.square(queue_head[:, :, 0]) + np.square(queue_head[:, :, 1]))

        # however, there is still a problem.  We should zero out the distance if the x position is within the bounds
        # of the pressure mat and the z is negative. This just indicates the person is pushing into the mat.
        queue_frame[:, :, 3] = (queue_frame[:, :, 2] - queue_frame[:, :, 0] - queue_frame[:, :, 1] * 1000000).clip(min=0)
        queue_head[:, :, 3] = (queue_head[:, :, 2] - queue_head[:, :, 0] - queue_frame[:, :, 1] * 1000000).clip(min=0)
        queue_frame[:, :, 2] = (queue_frame[:, :, 2] - queue_frame[:, :, 3]).clip(min=0)  # corrected Euclidean
        queue_head[:, :, 2] = (queue_head[:, :, 2] - queue_head[:, :, 3]).clip(min=0)  # corrected Euclidean

        # Now take the minimum of the Euclideans from the head and the frame planes
        distances[:, :] = np.amin([queue_frame[:, :, 2], queue_head[:, :, 2]], axis=0)


        return distances



    def get_penetration_weights(self, images, targets):

        try:
            bedangle = torch.mean(images[:, 2, 1:3, 0], dim=1)
            images = images.data.numpy()
            targets = targets.data.numpy()/1000
        except:
            bedangle = torch.mean(images[:, 2, 1:3, 0].cpu(), dim=1)
            images = images.cpu().data.numpy()
            targets = targets.cpu().data.numpy()/1000

        targets = np.reshape(targets, (targets.shape[0], targets.shape[1]/3, 3))



        queue_frame = np.zeros((targets.shape[0], targets.shape[1], 4))
        queue_head = np.zeros((targets.shape[0], targets.shape[1], 4))

        # get the shortest distance from the main frame of the bed. it's just the z.
        queue_frame[:, :, 0] = np.copy(targets[:, :, 2])

        # get the shortest distance from the head of the bed. you have to rotate about the bending point.
        By = (51) * 0.0286 - 0.0286 * 3 * np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis = 1)))
        queue_head[:, :, 0] = targets[:, :, 2] * np.cos(np.deg2rad(np.expand_dims(bedangle[:], axis = 1))) - (targets[:, :,1] - By) * np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis = 1) ))


        # now get the plane that the point is closer to.
        queue_frame[:, :, 2] += queue_frame[:, :, 0]
        queue_frame[:, :, 2][targets[:, :, 1] > 48*0.0286] = 0.0
        queue_head[:, :, 2] += queue_head[:, :, 0]
        queue_head[:, :, 2][targets[:, :, 1] <= 48*0.0286] = 0.0


        penetration_weights = queue_frame[:, :, 2] + queue_head[:, :, 2]
        penetration_weights[penetration_weights > 0] = 0.0
        penetration_weights[penetration_weights < 0] = 1.0


        # Get the distance off the side of the bed.  The bed is 27 pixels wide, so things over hanging this are added
        queue_frame[:, :, 1] = (-targets[:, :, 0] + 10 * 0.0286).clip(min=0)
        queue_frame[:, :, 1] = queue_frame[:, :, 1] + (targets[:, :, 0] - 37 * 0.0286).clip(min=0)
        queue_frame[:, :, 1] += 1.0
        queue_frame[:, :, 1][queue_frame[:, :, 1] > 1] = 0.0

        queue_head[:, :, 1] = np.copy(queue_frame[:, :, 1])  # the x distance does not depend on bed angle.

        penetration_weights = penetration_weights*queue_frame[:, :, 1]
        penetration_weights*=np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])*3
        penetration_weights+= 1


        return penetration_weights



    def get_penetration_distance(self, images, targets, bedangle = None):
        mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)

        try:
            images = images.data.numpy()
            targets = targets.data.numpy()/1000
            try:
                test = targets.shape[2]
            except:
                targets = np.reshape(targets, (targets.shape[0], targets.shape[1]/3, 3))
            bedangle = images[:, -1, 10, 10]


        except:
            images = np.expand_dims(images, axis = 0)
            targets = np.reshape(targets, (10, 3))
            targets = np.expand_dims(targets, axis = 0)
            bedangle = np.expand_dims(bedangle, axis = 0)



        distances = np.zeros((images.shape[0], targets.shape[1]))
        queue_frame = np.zeros((targets.shape[0], targets.shape[1], 4))
        queue_head = np.zeros((targets.shape[0], targets.shape[1], 4))

        # get the shortest distance from the main frame of the bed. it's just the z.
        queue_frame[:, :, 0] = targets[:, :, 2]

        # get the shortest distance from the head of the bed. you have to rotate about the bending point.


        By = (51) * 0.0286 - 0.0286 * 3 * np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis = 1)))
        queue_head[:, :, 0] = targets[:, :, 2] * np.cos(np.deg2rad(np.expand_dims(bedangle[:], axis = 1))) - (targets[:, :,1] - By) * np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis = 1) ))

        # Get the distance off the side of the bed.  The bed is 27 pixels wide, so things over hanging this are added
        queue_frame[:, :, 1] = (-targets[:, :, 0] + 10 * 0.0286).clip(min=0)
        queue_frame[:, :, 1] = queue_frame[:, :, 1] + (targets[:, :, 0] - 37 * 0.0286).clip(min=0)
        queue_head[:, :, 1] = np.copy(queue_frame[:, :, 1])  # the x distance does not depend on bed angle.

        # Now take the Euclidean for each frame and head set
        queue_frame[:, :, 2] = np.sqrt(np.square(queue_frame[:, :, 0]) + np.square(queue_frame[:, :, 1]))
        queue_head[:, :, 2] = np.sqrt(np.square(queue_head[:, :, 0]) + np.square(queue_head[:, :, 1]))

        # however, there is still a problem.  We should zero out the distance if the x position is within the bounds
        # of the pressure mat and the z is negative. This just indicates the person is pushing into the mat.
        queue_frame[:, :, 3] = (queue_frame[:, :, 2] - queue_frame[:, :, 0] - queue_frame[:, :, 1] * 1000000).clip(min=0)
        queue_head[:, :, 3] = (queue_head[:, :, 2] - queue_head[:, :, 0] - queue_frame[:, :, 1] * 1000000).clip(min=0)
        queue_frame[:, :, 2] = (queue_frame[:, :, 2] - queue_frame[:, :, 3]).clip(min=0)  # corrected Euclidean
        queue_head[:, :, 2] = (queue_head[:, :, 2] - queue_head[:, :, 3]).clip(min=0)  # corrected Euclidean

        # Now take the minimum of the Euclideans from the head and the frame planes
        distances[:, :] = np.amin([queue_frame[:, :, 2], queue_head[:, :, 2]], axis=0)


        return distances




    def forward_upper_kinematics(self, images, torso_lengths, angles):
        #print images.shape, 'images shape'
        #print torso_lengths.shape, 'torso lengths shape'
        #print angles.shape, 'angles shape'


        try: #this happens when we call it from the convnet which has a tensor var we need to convert
            images = images.data.numpy()
            torso_lengths = torso_lengths.data.numpy()
            angles = angles.data.numpy()
            lengths = torso_lengths[:, 3:12] / 100
            angles = angles[:, 0:10]
            angles[:, 2:4] = angles[:, 2:4] * 0.25
            torso = torso_lengths[:, 0:3] / 100

        except: #this happens when we call it from create dataset and sent it a numpy array instead of a tensor var
            lengths = np.expand_dims(torso_lengths[3:12], axis = 0)
            angles = np.expand_dims(angles[0:10], axis = 0)
            torso = np.expand_dims(torso_lengths[0:3], axis = 0)


        #print lengths[0,:], angles[0,:], torso[0,:]
        targets = np.zeros((images.shape[0], 18))
        queue = np.zeros((6,3))
        for set in range(0, images.shape[0]):
            try: #this happens when the images are actually the images
                bedangle = images[set, -1, 10, 10]
            except: #this happens when you just throw an angle in there
                bedangle = images
            TrelO = tft.identity_matrix()
            TprelT = tft.identity_matrix()

            NrelTp = tft.rotation_matrix(np.deg2rad(bedangle * 0.75), (1, 0, 0))

            TrelO[0:3, 3] = torso[set, :].T
            TprelT[2, 3] = -lengths[set, 0]
            NrelTp[1, 3] = lengths[set, 1] * np.cos(np.deg2rad(bedangle * 0.75))
            NrelTp[2, 3] = lengths[set, 1] * np.sin(np.deg2rad(bedangle * 0.75))

            rSrelN = tft.identity_matrix()
            rSrelN[0, 3] = -lengths[set, 2]

            lSrelN = tft.identity_matrix()
            lSrelN[0, 3] = lengths[set, 2]



            Pr_NS = np.array([[-lengths[set, 2]], [0], [0], [1]])
            Pl_NS = np.array([[lengths[set, 2]], [0], [0], [1]])

            HrelN = np.matmul(tft.rotation_matrix(np.deg2rad(-angles[set, 8] + 90), (0, 0, 1)),
                                          tft.rotation_matrix(np.deg2rad(angles[set, 9] - 90), (0, 1, 0)))
            rErelrS = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(-angles[set, 4] + 180), (0, 1, 0)),
                                          tft.rotation_matrix(np.deg2rad(180 + angles[set, 2]), (0, 0, 1))),
                                tft.rotation_matrix(np.deg2rad((angles[set, 0]) + 90 + angles[set, 4]), (-1, 0, 0)))
            lErellS = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(angles[set, 5] + 180), (0, 1, 0)),
                                          tft.rotation_matrix(np.deg2rad(-angles[set, 3]), (0, 0, 1))),
                                tft.rotation_matrix(np.deg2rad((-angles[set, 1]) + 90 - angles[set, 5]), (-1, 0, 0)))


            P_NH = np.matmul(HrelN, np.array([[lengths[set, 8]], [0], [0], [1]]))

            #print angles[set, 8:10]
            #print lengths[set, 8]
            #print P_NH,'Pnh'

            Pr_SE = np.matmul(rErelrS, np.array([[lengths[set, 4]], [0], [0], [1]]))
            Pl_SE = np.matmul(lErellS, np.array([[lengths[set, 5]], [0], [0], [1]]))



            HrelN[0:3, 3] = -P_NH[0:3, 0]
            rErelrS[0:3, 3] = -Pr_SE[0:3, 0]
            lErellS[0:3, 3] = -Pl_SE[0:3, 0]

            # rHrelrE = np.matmul(tft.rotation_matrix(np.deg2rad(-(angles[0])), (-1,0,0)),tft.rotation_matrix(np.deg2rad(angles[6]), (0, 0, 1)))
            rHrelrE = tft.rotation_matrix(np.deg2rad(angles[set, 6]), (0, 0, 1))
            lHrellE = tft.rotation_matrix(np.deg2rad(angles[set, 7]), (0, 0, 1))

            # print rHrelrE, 'rhrelre'
            # print lHrellE, 'lhrelle'

            Pr_EH = np.matmul(rHrelrE, np.array([[lengths[set, 6]], [0], [0], [1]]))
            Pl_EH = np.matmul(lHrellE, np.array([[lengths[set, 7]], [0], [0], [1]]))

            Pr_SE = -Pr_SE
            Pr_SE[3, 0] = 1
            Pl_SE = -Pl_SE
            Pl_SE[3, 0] = 1


            pred_H = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), P_NH))
            queue[0, :] = np.squeeze(pred_H[0:3, 0].T)

            queue[1, :] = torso[set, :]

            pred_r_S = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), Pr_NS))
            #print pred_r_S, 'pred r S'

            pred_l_S = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), Pl_NS))
            #print pred_l_S, 'pred l S'

            pred_r_E = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), rSrelN), Pr_SE))
            queue[2, :] = np.squeeze(pred_r_E[0:3, 0].T)

            pred_l_E = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), lSrelN), Pl_SE))
            queue[3, :] = np.squeeze(pred_l_E[0:3, 0].T)

            pred_r_H = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), rSrelN), rErelrS), Pr_EH))
            queue[4, :] = np.squeeze(pred_r_H[0:3, 0].T)

            pred_l_H = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), lSrelN), lErellS), Pl_EH))
            queue[5, :] = np.squeeze(pred_l_H[0:3, 0].T)
            #print targets, 'targets'

            targets[set, :] = queue.flatten()

            #print targets[set, :], 'target set'

        targets = targets*1000
        return targets


    def forward_lower_kinematics(self, images, torso_lengths, angles):
        #print images.shape, 'images shape'
        #print torso_lengths.shape, 'torso lengths shape'
        #print angles.shape, 'angles shape'


        try: #this happens when we call it from the convnet which has a tensor var we need to convert
            images = images.data.numpy()
            torso_lengths = torso_lengths.data.numpy()
            angles = angles.data.numpy()
            lengths = torso_lengths[:, 3:11] / 100
            angles = angles[:, 0:8]
            angles[:, 2:4] = angles[:, 2:4] * 0.25
            torso = torso_lengths[:, 0:3] / 100

        except: #this happens when we call it from create dataset and sent it a numpy array instead of a tensor var
            lengths = np.expand_dims(torso_lengths[3:11], axis = 0)
            angles = np.expand_dims(angles[0:8], axis = 0)
            torso = np.expand_dims(torso_lengths[0:3], axis = 0)


        #print lengths[0,:], angles[0,:], torso[0,:]
        targets = np.zeros((images.shape[0], 15))
        queue = np.zeros((5,3))
        for set in range(0, images.shape[0]):
            try: #this happens when the images are actually the images
                bedangle = images[set, -1, 10, 10]
            except: #this happens when you just throw an angle in there
                bedangle = images
            TrelO = tft.identity_matrix()
            TprelT = tft.identity_matrix()

            BrelTp = tft.identity_matrix()

            TrelO[0:3, 3] = torso[set, :].T
            TprelT[2, 3] = -lengths[set, 0]
            BrelTp[1, 3] = -lengths[set, 1]


            rGrelB = tft.identity_matrix()
            rGrelB[0, 3] = -lengths[set, 2]

            lGrelB = tft.identity_matrix()
            lGrelB[0, 3] = lengths[set, 2]



            Pr_BG = np.array([[-lengths[set, 2]], [0], [0], [1]])
            Pl_BG = np.array([[lengths[set, 2]], [0], [0], [1]])


            rKrelrG = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(angles[set, 4]), (0, 1, 0)),
                                          tft.rotation_matrix(np.deg2rad(180 + angles[set, 2]), (0, 0, 1))),
                                tft.rotation_matrix(np.deg2rad((-angles[set, 0]) + 90 + angles[set, 4]), (-1, 0, 0)))
            lKrellG = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(angles[set, 5] - 180), (0, 1, 0)),
                                          tft.rotation_matrix(np.deg2rad(-angles[set, 3]), (0, 0, 1))),
                                tft.rotation_matrix(np.deg2rad((-angles[set, 1]) + 90 - angles[set, 5]), (-1, 0, 0)))


            Pr_GK = np.matmul(rKrelrG, np.array([[lengths[set, 4]], [0], [0], [1]]))
            Pl_GK = np.matmul(lKrellG, np.array([[lengths[set, 5]], [0], [0], [1]]))

            rKrelrG[0:3, 3] = -Pr_GK[0:3, 0]
            lKrellG[0:3, 3] = -Pl_GK[0:3, 0]

            # rHrelrE = np.matmul(tft.rotation_matrix(np.deg2rad(-(angles[0])), (-1,0,0)),tft.rotation_matrix(np.deg2rad(angles[6]), (0, 0, 1)))
            rArelrK = tft.rotation_matrix(np.deg2rad(angles[set, 6]), (0, 0, 1))
            lArellK = tft.rotation_matrix(np.deg2rad(angles[set, 7]), (0, 0, 1))

            # print rHrelrE, 'rhrelre'
            # print lHrellE, 'lhrelle'

            Pr_KA = np.matmul(rArelrK, np.array([[lengths[set, 6]], [0], [0], [1]]))
            Pl_KA = np.matmul(lArellK, np.array([[lengths[set, 7]], [0], [0], [1]]))

            Pr_GK = -Pr_GK
            Pr_GK[3, 0] = 1
            Pl_GK = -Pl_GK
            Pl_GK[3, 0] = 1

            queue[0, :] = torso[set, :]

            pred_r_G = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), Pr_BG))
            #print pred_r_G, 'pred r G'

            pred_l_G = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), Pl_BG))
            #print pred_l_G, 'pred l G'

            pred_r_K = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), rGrelB), Pr_GK))
            queue[1, :] = np.squeeze(pred_r_K[0:3, 0].T)

            pred_l_K = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), lGrelB), Pl_GK))
            queue[2, :] = np.squeeze(pred_l_K[0:3, 0].T)
            #print pred_l_K, 'pred_l_K'

            pred_r_A = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), rGrelB), rKrelrG), Pr_KA))
            queue[3, :] = np.squeeze(pred_r_A[0:3, 0].T)

            pred_l_A = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), lGrelB), lKrellG), Pl_KA))
            queue[4, :] = np.squeeze(pred_l_A[0:3, 0].T)
            #print targets, 'targets'

            targets[set, :] = queue.flatten()

            #print targets[set, :], 'target set'

        targets = targets*1000
        return targets



    def forward_kinematics_pytorch_R(self, images_v, torso_lengths_angles_v, loss_vector_type, targets_v=None,  kincons_v = None, forward_only = False, count = 500):

        test_ground_truth = False
        pseudotargets = None

        if loss_vector_type == 'anglesCL' or loss_vector_type == 'anglesVL':
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = F.pad(torso_lengths_angles_v, (0, 27, 0, 0)) #make more room for head, arms, and legs x, y, z coords.  torso already is in the network.
            # print torso_lengths_angles_v.size()

            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)

            if test_ground_truth == True:
                # print kincons_v.size()
                torso_lengths_angles_v[:, 0:18] = kincons_v[:, 0:18] #this is the upper angles, lower angles, upper lengths, lower lengths in that order
                torso_lengths_angles_v[:, 20:37] = kincons_v[:, 18:35]
                torso_lengths_angles_v[:, 37:40] = targets_v[:, 3:6] / 1000 #this is the torso x, y, z coords
                # print targets_v[0, :], 'targets'
                # print torso_lengths_angles_v[0, :]

            # images = images_v.data.numpy() * np.pi / 180
            images = images_v.data * np.pi / 180
            bedangle = images[:, -1, 10, 10] * 0.75

            bedangle = Variable(bedangle)


            #print torso_lengths_angles_v[0, 17], torso_lengths_angles_v[0, 18], torso_lengths_angles_v.size()

            torso_lengths_angles_v[:, 0] = torch.clamp(torso_lengths_angles_v[:, 0], -1.8, 1.8)
            torso_lengths_angles_v[:, 1] = torch.clamp(torso_lengths_angles_v[:, 1], -1.8, 1.8)
            torso_lengths_angles_v[:, 2] = torch.clamp(torso_lengths_angles_v[:, 2], -1.35, 1.35)
            torso_lengths_angles_v[:, 3] = torch.clamp(torso_lengths_angles_v[:, 3], -1.35, 1.35)
            torso_lengths_angles_v[:, 4] = torch.clamp(torso_lengths_angles_v[:, 4], -1.35, 1.35)
            torso_lengths_angles_v[:, 5] = torch.clamp(torso_lengths_angles_v[:, 5], -1.35, 1.35)
            torso_lengths_angles_v[:, 6] = torch.clamp(torch.add(torso_lengths_angles_v[:, 6], 1.5), 0.2, 1.8)
            torso_lengths_angles_v[:, 7] = torch.clamp(torch.add(torso_lengths_angles_v[:, 7], 1.5), 0.2, 1.8)
            #torso_lengths_angles_v[:, 8] = torch.clamp(torso_lengths_angles_v[:, 6], -1.8, 1.8)
            #torso_lengths_angles_v[:, 9] = torch.clamp(torso_lengths_angles_v[:, 7], -1.5, 1.5)
            torso_lengths_angles_v[:, 10] = torch.clamp(torso_lengths_angles_v[:, 10], -1.8, 1.8)
            torso_lengths_angles_v[:, 11] = torch.clamp(torso_lengths_angles_v[:, 11], -1.8, 1.8)
            torso_lengths_angles_v[:, 12] = torch.clamp(torch.add(torso_lengths_angles_v[:, 12], -0.6), -1.8, 0.)
            torso_lengths_angles_v[:, 13] = torch.clamp(torch.add(torso_lengths_angles_v[:, 13], -0.6), -1.8, 0.)
            torso_lengths_angles_v[:, 14] = torch.clamp(torch.add(torso_lengths_angles_v[:, 14], -0.6), -1.35, 1.35)
            torso_lengths_angles_v[:, 15] = torch.clamp(torch.add(torso_lengths_angles_v[:, 15], -0.6), -1.35, 1.35)
            torso_lengths_angles_v[:, 16] = torch.clamp(torch.add(torso_lengths_angles_v[:, 16], 1.5), 0.2, 1.8)
            torso_lengths_angles_v[:, 17] = torch.clamp(torch.add(torso_lengths_angles_v[:, 17], 1.5), 0.2, 1.8)
            torso_lengths_angles_v[:, 18] = torch.clamp(torch.add(torso_lengths_angles_v[:, 18], 0.2), -0.5, 1.3) #torso angle for upper
            torso_lengths_angles_v[:, 19] = torch.clamp(torch.add(torso_lengths_angles_v[:, 19], 0.), -1.3, 0.5) #torso angle for lower




            if True:#scount > 300 and loss_vector_type == 'anglesCL':  # add this bit for constant bone lengths
                # if subject is not None:
                torso_lengths_angles = Variable(torso_lengths_angles_v.data.clone())

                torso_lengths_angles = torso_lengths_angles.data
                print subject, 'CONSTANT BONE LENGTHS, SUBJECT ', str(subject)
                if subject == 9 or subject == 4:
                    torso_lengths_angles[:, 20:37] = torch.Tensor(
                        [0.1, 0.26832222, 0.17381444, 0.17381444, 0.28164876, 0.27695534, 0.21541507, 0.20452102,
                         0.31553109, 0.14, 0.20127556, 0.10428444, 0.10428444, 0.41213504, 0.43190713, 0.42485215,
                         0.41069972]).repeat(torso_lengths_angles.size()[0], 1)
                elif subject == 10:
                    torso_lengths_angles[:, 20:37] = torch.Tensor(
                        [0.1, 0.27084611, 0.17545882, 0.17545882, 0.2905044, 0.28538682, 0.2236274, 0.21349257,
                         0.30564677, 0.14, 0.20316878, 0.10527102, 0.10527102, 0.43162535, 0.44191622, 0.42166114,
                         0.41218717]).repeat(torso_lengths_angles.size()[0], 1)
                elif subject == 11:
                    torso_lengths_angles[:, 20:37] = torch.Tensor(
                        [0.1, 0.26946944, 0.17456189, 0.17456189, 0.28605329, 0.28458026, 0.22021226, 0.20930997,
                         0.30607809, 0.14, 0.20213611, 0.10473289, 0.10473289, 0.42447277, 0.43699407, 0.42203217,
                         0.41353693]).repeat(torso_lengths_angles.size()[0], 1)
                elif subject == 12:
                    torso_lengths_angles[:, 20:37] = torch.Tensor(
                        [0.1, 0.27382889, 0.17740218, 0.17740218, 0.29012795, 0.28606032, 0.22466585, 0.21228756,
                         0.30998514, 0.14, 0.20540622, 0.10643698, 0.10643698, 0.4223127, 0.43911505, 0.43320225,
                         0.41609839]).repeat(torso_lengths_angles.size()[0], 1)
                elif subject == 13:
                    torso_lengths_angles[:, 20:37] = torch.Tensor(
                        [0.1, 0.27084611, 0.17545882, 0.17545882, 0.28455711, 0.28242044, 0.21731193, 0.20271964,
                         0.31538197, 0.14, 0.20316878, 0.10527102, 0.10527102, 0.4135731, 0.42983224, 0.42582947,
                         0.41555586]).repeat(torso_lengths_angles.size()[0], 1)
                elif subject == 14:
                    torso_lengths_angles[:, 20:37] = torch.Tensor(
                        [0.1, 0.26878111, 0.17411342, 0.17411342, 0.2839482, 0.28109441, 0.21763367, 0.20413045,
                         0.31335409, 0.14, 0.20161978, 0.10446382, 0.10446382, 0.41581538, 0.42474949, 0.42284857,
                         0.41566802]).repeat(torso_lengths_angles.size()[0], 1)
                elif subject == 15:
                    torso_lengths_angles[:, 20:37] = torch.Tensor(
                        [0.1, 0.26969889, 0.17471138, 0.17471138, 0.28589227, 0.28515002, 0.22193393, 0.20341761,
                         0.30919443, 0.14, 0.20230822, 0.10482258, 0.10482258, 0.42598388, 0.42974338, 0.42117544,
                         0.41946262]).repeat(torso_lengths_angles.size()[0], 1)
                elif subject == 16:
                    torso_lengths_angles[:, 20:37] = torch.Tensor(
                        [0.1, 0.27314056, 0.17695371, 0.17695371, 0.28672228, 0.28045977, 0.22281829, 0.20828243,
                         0.31858438, 0.14, 0.20488989, 0.10616791, 0.10616791, 0.42033598, 0.43259464, 0.4245889,
                         0.41915392]).repeat(torso_lengths_angles.size()[0], 1)
                elif subject == 17:
                    torso_lengths_angles[:, 20:37] = torch.Tensor(
                        [0.1, 0.27428778, 0.17770116, 0.17770116, 0.28976403, 0.28482488, 0.22293538, 0.20923963,
                         0.3152724, 0.14, 0.20575044, 0.10661636, 0.10661636, 0.41702271, 0.42695342, 0.43464569,
                         0.42858238]).repeat(torso_lengths_angles.size()[0], 1)
                elif subject == 18:
                    torso_lengths_angles[:, 20:37] = torch.Tensor(
                        [0.1, 0.27176389, 0.17605678, 0.17605678, 0.28688743, 0.28375046, 0.223929, 0.21221132,
                         0.31124915, 0.14, 0.20385722, 0.10562978, 0.10562978, 0.42229295, 0.43479301, 0.42381791,
                         0.41654181]).repeat(torso_lengths_angles.size()[0], 1)
                torso_lengths_angles = Variable(torso_lengths_angles)


            else:
                print 'VARIABLE BONE LENGTHS, PULL OUT, SUBJECT ', str(subject)
                torso_lengths_angles = Variable(torso_lengths_angles_v.data)
                torso_lengths_angles_v[:, 20] = torch.add(torso_lengths_angles_v[:, 20], 0.1)
                torso_lengths_angles_v[:, 21] = torch.add(torso_lengths_angles_v[:, 21], 0.26)
                torso_lengths_angles_v[:, 22] = torch.add(torso_lengths_angles_v[:, 22], 0.17)
                torso_lengths_angles_v[:, 23] = torch.add(torso_lengths_angles_v[:, 23], 0.17)
                torso_lengths_angles_v[:, 24] = torch.add(torso_lengths_angles_v[:, 24], 0.28)
                torso_lengths_angles_v[:, 25] = torch.add(torso_lengths_angles_v[:, 25], 0.28)
                torso_lengths_angles_v[:, 26] = torch.add(torso_lengths_angles_v[:, 26], 0.19)
                torso_lengths_angles_v[:, 27] = torch.add(torso_lengths_angles_v[:, 27], 0.19)
                torso_lengths_angles_v[:, 28] = torch.add(torso_lengths_angles_v[:, 28], 0.28)
                torso_lengths_angles_v[:, 29] = torch.add(torso_lengths_angles_v[:, 29], 0.14)
                torso_lengths_angles_v[:, 30] = torch.add(torso_lengths_angles_v[:, 30], 0.19)
                torso_lengths_angles_v[:, 31] = torch.add(torso_lengths_angles_v[:, 31], 0.10)
                torso_lengths_angles_v[:, 32] = torch.add(torso_lengths_angles_v[:, 32], 0.10)
                torso_lengths_angles_v[:, 33] = torch.add(torso_lengths_angles_v[:, 33], 0.40)
                torso_lengths_angles_v[:, 34] = torch.add(torso_lengths_angles_v[:, 34], 0.40)
                torso_lengths_angles_v[:, 35] = torch.add(torso_lengths_angles_v[:, 35], 0.30)
                torso_lengths_angles_v[:, 36] = torch.add(torso_lengths_angles_v[:, 36], 0.30)


            torso_lengths_angles_v[:, 37] = torch.add(torso_lengths_angles_v[:, 37], 0.6)
            torso_lengths_angles_v[:, 38] = torch.add(torso_lengths_angles_v[:, 38], 1.3)
            torso_lengths_angles_v[:, 39] = torch.add(torso_lengths_angles_v[:, 39], 0.1)


            #head in vectorized form
            torso_lengths_angles_v[:, 40] = torso_lengths_angles_v[:, 37] \
                                            + torso_lengths_angles[:, 28] * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).cos()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos())
            torso_lengths_angles_v[:, 41] = torso_lengths_angles_v[:, 38] + torso_lengths_angles[:, 28] * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).sin()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).cos() + torso_lengths_angles[:, 28] * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
            torso_lengths_angles_v[:, 42] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + torso_lengths_angles[:, 28] * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).sin()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).sin() - torso_lengths_angles[:, 28] * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

            # right elbow in vectorized form
            torso_lengths_angles_v[:, 43] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 22] \
                                            + (-(np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 24] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos())
            torso_lengths_angles_v[:, 44] = torso_lengths_angles_v[:, 38] \
                                            + (-torso_lengths_angles[:, 24] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            - ((np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 24] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
            torso_lengths_angles_v[:, 45] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] \
                                            + (-torso_lengths_angles[:, 24] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + ((np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 24] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

            # left elbow in vectorized form
            torso_lengths_angles_v[:, 46] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 23] \
                                            + (-(np.pi + torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 25] * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos())
            torso_lengths_angles_v[:, 47] = torso_lengths_angles_v[:, 38] \
                                            + (-torso_lengths_angles[:, 25] * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + ((np.pi + torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 25] * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
            torso_lengths_angles_v[:, 48] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] \
                                            + (-torso_lengths_angles[:, 25] * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            - ((np.pi + torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 25] * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

            # right hand in vectorized form
            torso_lengths_angles_v[:, 49] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 22] \
                                            + ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 26] - torso_lengths_angles[:, 25]) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 26]) \
                                            + ((1.8 -torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 26]
            torso_lengths_angles_v[:, 50] = torso_lengths_angles_v[:, 38] \
                                            + (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 26] - torso_lengths_angles[:, 25]) + ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 26]) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            - (((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 26] - torso_lengths_angles[:, 25]) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 26]) - ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 26]) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
            torso_lengths_angles_v[:, 51] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20]\
                                            + (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 26] - torso_lengths_angles[:, 25]) + ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 26]) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + (((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 26] - torso_lengths_angles[:, 25]) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 26]) - ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 26]) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

            # left hand in vectorized form
            torso_lengths_angles_v[:, 52] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 23] \
                                            + ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 27] - torso_lengths_angles[:, 25]) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 27]) \
                                            + ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 27]
            torso_lengths_angles_v[:, 53] = torso_lengths_angles_v[:, 38] \
                                            + (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 27] - torso_lengths_angles[:, 25]) + ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 27]) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            - (((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 27] - torso_lengths_angles[:, 25]) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 27]) - ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 27]) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
            torso_lengths_angles_v[:, 54] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20]\
                                            + (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 27] - torso_lengths_angles[:, 25]) + ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 27]) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + (((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 27] - torso_lengths_angles[:, 25]) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 27]) - ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 27]) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()


            # right knee in vectorized form
            torso_lengths_angles_v[:, 55] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 31] \
                                            + ((torso_lengths_angles_v[:, 14] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).cos())
            torso_lengths_angles_v[:, 56] = torso_lengths_angles_v[:, 38] \
                                            + (-torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            - ((torso_lengths_angles_v[:, 14] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
            torso_lengths_angles_v[:, 57] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] \
                                            + (-torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            + ((torso_lengths_angles_v[:, 14] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()

            # left knee in vectorized form
            torso_lengths_angles_v[:, 58] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 32] \
                                            + (-((-1.8 - torso_lengths_angles_v[:, 15]) * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 34] * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).cos())
            torso_lengths_angles_v[:, 59] = torso_lengths_angles_v[:, 38] \
                                            + (-torso_lengths_angles[:, 34] * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + (((-1.8 - torso_lengths_angles_v[:, 15]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 34] * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
            torso_lengths_angles_v[:, 60] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] \
                                            + (-torso_lengths_angles[:, 34] * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - (((-1.8 - torso_lengths_angles_v[:, 15]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 34] * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()

            # right ankle in vectorized form
            torso_lengths_angles_v[:, 61] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 31] \
                                            - (-(torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).cos() * (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 35] -  torso_lengths_angles[:, 33]) - ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 35]) \
                                            + (-(torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]
            torso_lengths_angles_v[:, 62] = torso_lengths_angles_v[:, 38] \
                                            + (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 35] - torso_lengths_angles[:, 33]) + ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 35]) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + (((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 35] -  torso_lengths_angles[:, 33]) - ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]) - ((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
            torso_lengths_angles_v[:, 63] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] \
                                            + (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 35] - torso_lengths_angles[:, 33]) + ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 35]) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - (((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 35] -  torso_lengths_angles[:, 33]) - ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]) - ((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()

            # left ankle in vectorized form, need to fix third line of [:,65]
            torso_lengths_angles_v[:, 64] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 32] \
                                            + (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).cos() * (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 36] - torso_lengths_angles[:, 34]) - ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 36]) \
                                            + (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).sin() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 36]
            torso_lengths_angles_v[:, 65] = torso_lengths_angles_v[:, 38] \
                                            + (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 36] - torso_lengths_angles[:, 34]) + ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) * (0. + torso_lengths_angles_v[:, 19]).cos()\
                                            - ((((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 36] - torso_lengths_angles[:, 34]) - ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) - (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).sin() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
            torso_lengths_angles_v[:, 66] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] \
                                            + (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 36] - torso_lengths_angles[:, 34]) + ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            + ((((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 36] - torso_lengths_angles[:, 34]) - ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) - (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).sin() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()



            if forward_only == True:
                #let's get the neck, shoulders, and glutes pseudotargets
                pseudotargets = Variable(torch.Tensor(np.zeros((images.shape[0], 15))))

                #get the neck in vectorized form
                pseudotargets[:, 0] = torso_lengths_angles_v[:, 37]
                pseudotargets[:, 1] = torso_lengths_angles_v[:, 38] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                pseudotargets[:, 2] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

                #get the right shoulder in vectorized form
                pseudotargets[:, 3] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 22]
                pseudotargets[:, 4] = torso_lengths_angles_v[:, 38] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                pseudotargets[:, 5] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

                #print \the left shoulder in vectorized form
                pseudotargets[:, 6] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 23]
                pseudotargets[:, 7] = torso_lengths_angles_v[:, 38] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                pseudotargets[:, 8] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

                #get the right glute in vectorized form
                pseudotargets[:, 9] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 31]
                pseudotargets[:, 10] = torso_lengths_angles_v[:, 38] - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
                pseudotargets[:, 11] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()

                #print \the left glute in vectorized form
                pseudotargets[:, 12] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 32]
                pseudotargets[:, 13] = torso_lengths_angles_v[:, 38] - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
                pseudotargets[:, 14] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()

                pseudotargets = pseudotargets.data.numpy() * 1000

            # angles = torso_lengths_angles_v[:, 0:20].data.numpy()*100
            angles = torso_lengths_angles[:, 0:20]*100
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = F.pad(torso_lengths_angles_v, (-20, 0, 0, 0)) #cut off all the angles
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)


        return torso_lengths_angles_v, angles, pseudotargets



    def forward_kinematics_lengthsv_pytorch(self, images_v, torso_lengths_angles_v, loss_vector_type, kincons_v = None, forward_only = False, subject = None):

        test_ground_truth = False
        pseudotargets = None

        if loss_vector_type == 'anglesSTVL':
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = F.pad(torso_lengths_angles_v, (0, 27, 0, 0)) #make more room for head, arms, and legs x, y, z coords.  torso already is in the network.
            # print torso_lengths_angles_v.size()

            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)

            if test_ground_truth == True:
                # print kincons_v.size()
                torso_lengths_angles_v[:, 0:18] = kincons_v[:, 0:18] #this is the upper angles, lower angles, upper lengths, lower lengths in that order
                torso_lengths_angles_v[:, 20:37] = kincons_v[:, 18:35]
                torso_lengths_angles_v[:, 37:40] = targets_v[:, 3:6] / 1000 #this is the torso x, y, z coords
                # print targets_v[0, :], 'targets'
                # print torso_lengths_angles_v[0, :]

            # images = images_v.data.numpy() * np.pi / 180
            images = images_v.data * np.pi / 180
            bedangle = images[:, -1, 10, 10] * 0.75


            torso_lengths_angles = Variable(torso_lengths_angles_v.data)
            bedangle = Variable(bedangle)

            torso_lengths_angles_v[:, 0] = torch.clamp(torso_lengths_angles_v[:, 0], -1.8, 1.8)
            torso_lengths_angles_v[:, 1] = torch.clamp(torso_lengths_angles_v[:, 1], -1.8, 1.8)
            torso_lengths_angles_v[:, 2] = torch.clamp(torso_lengths_angles_v[:, 2], -1.35, 1.35)
            torso_lengths_angles_v[:, 3] = torch.clamp(torso_lengths_angles_v[:, 3], -1.35, 1.35)
            torso_lengths_angles_v[:, 4] = torch.clamp(torso_lengths_angles_v[:, 4], -1.35, 1.35)
            torso_lengths_angles_v[:, 5] = torch.clamp(torso_lengths_angles_v[:, 5], -1.35, 1.35)
            torso_lengths_angles_v[:, 6] = torch.clamp(torch.add(torso_lengths_angles_v[:, 6], 1.5), 0.2, 1.8)
            torso_lengths_angles_v[:, 7] = torch.clamp(torch.add(torso_lengths_angles_v[:, 7], 1.5), 0.2, 1.8)
            #torso_lengths_angles_v[:, 8] = torch.clamp(torso_lengths_angles_v[:, 6], -1.8, 1.8)
            #torso_lengths_angles_v[:, 9] = torch.clamp(torso_lengths_angles_v[:, 7], -1.5, 1.5)
            torso_lengths_angles_v[:, 10] = torch.clamp(torso_lengths_angles_v[:, 10], -1.8, 1.8)
            torso_lengths_angles_v[:, 11] = torch.clamp(torso_lengths_angles_v[:, 11], -1.8, 1.8)
            torso_lengths_angles_v[:, 12] = torch.clamp(torch.add(torso_lengths_angles_v[:, 12], -0.6), -1.8, 0.)
            torso_lengths_angles_v[:, 13] = torch.clamp(torch.add(torso_lengths_angles_v[:, 13], -0.6), -1.8, 0.)
            torso_lengths_angles_v[:, 14] = torch.clamp(torch.add(torso_lengths_angles_v[:, 14], -0.6), -1.35, 1.35)
            torso_lengths_angles_v[:, 15] = torch.clamp(torch.add(torso_lengths_angles_v[:, 15], -0.6), -1.35, 1.35)
            torso_lengths_angles_v[:, 16] = torch.clamp(torch.add(torso_lengths_angles_v[:, 16], 1.5), 0.2, 1.8)
            torso_lengths_angles_v[:, 17] = torch.clamp(torch.add(torso_lengths_angles_v[:, 17], 1.5), 0.2, 1.8)
            torso_lengths_angles_v[:, 18] = torch.clamp(torch.add(torso_lengths_angles_v[:, 18], 0.2), -0.5, 1.3) #torso angle for upper
            torso_lengths_angles_v[:, 19] = torch.clamp(torch.add(torso_lengths_angles_v[:, 19], 0.), -1.3, 0.5) #torso angle for lower

            print 'VARIABLE BONE LENGTHS, STRAIGHT THROUGH, SUBJECT ', str(subject)

            torso_lengths_angles_v[:, 20] = torch.add(torso_lengths_angles_v[:, 20], 0.1)
            torso_lengths_angles_v[:, 21] = torch.add(torso_lengths_angles_v[:, 21], 0.26)
            torso_lengths_angles_v[:, 22] = torch.add(torso_lengths_angles_v[:, 22], 0.17)
            torso_lengths_angles_v[:, 23] = torch.add(torso_lengths_angles_v[:, 23], 0.17)
            torso_lengths_angles_v[:, 24] = torch.add(torso_lengths_angles_v[:, 24], 0.28)
            torso_lengths_angles_v[:, 25] = torch.add(torso_lengths_angles_v[:, 25], 0.28)
            torso_lengths_angles_v[:, 26] = torch.add(torso_lengths_angles_v[:, 26], 0.19)
            torso_lengths_angles_v[:, 27] = torch.add(torso_lengths_angles_v[:, 27], 0.19)
            torso_lengths_angles_v[:, 28] = torch.add(torso_lengths_angles_v[:, 28], 0.28)
            torso_lengths_angles_v[:, 29] = torch.add(torso_lengths_angles_v[:, 29], 0.14)
            torso_lengths_angles_v[:, 30] = torch.add(torso_lengths_angles_v[:, 30], 0.19)
            torso_lengths_angles_v[:, 31] = torch.add(torso_lengths_angles_v[:, 31], 0.10)
            torso_lengths_angles_v[:, 32] = torch.add(torso_lengths_angles_v[:, 32], 0.10)
            torso_lengths_angles_v[:, 33] = torch.add(torso_lengths_angles_v[:, 33], 0.40)
            torso_lengths_angles_v[:, 34] = torch.add(torso_lengths_angles_v[:, 34], 0.40)
            torso_lengths_angles_v[:, 35] = torch.add(torso_lengths_angles_v[:, 35], 0.30)
            torso_lengths_angles_v[:, 36] = torch.add(torso_lengths_angles_v[:, 36], 0.30)
            torso_lengths_angles_v[:, 37] = torch.add(torso_lengths_angles_v[:, 37], 0.6)
            torso_lengths_angles_v[:, 38] = torch.add(torso_lengths_angles_v[:, 38], 1.3)
            torso_lengths_angles_v[:, 39] = torch.add(torso_lengths_angles_v[:, 39], 0.1)

            #print torso_lengths_angles_v[0, 20:39], 'LENGTHS!'





            #head in vectorized form
            torso_lengths_angles_v[:, 40] = torso_lengths_angles_v[:, 37] \
                                            + (0.+torso_lengths_angles_v[:, 28]) * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).cos()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos())
            torso_lengths_angles_v[:, 41] = torso_lengths_angles_v[:, 38] + (0.+torso_lengths_angles_v[:, 28]) * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).sin()) * (( -np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:,18]).cos() + (0.+torso_lengths_angles_v[:, 28]) * ((-np.pi / 2. + torso_lengths_angles_v[:,9] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + (0.+torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).cos()
            torso_lengths_angles_v[:, 42] = torso_lengths_angles_v[:, 39] - (0.+torso_lengths_angles_v[:, 20]) + (0.+torso_lengths_angles_v[:, 28]) * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).sin()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).sin() - (0.+torso_lengths_angles_v[:, 28]) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + (0.+torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).sin()

            # right elbow in vectorized form
            torso_lengths_angles_v[:, 43] = torso_lengths_angles_v[:, 37] - (0.+torso_lengths_angles_v[:, 22]) \
                                            + (-(np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 24]) * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos())
            torso_lengths_angles_v[:, 44] = torso_lengths_angles_v[:, 38] \
                                            + (-(0.+torso_lengths_angles_v[:, 24]) * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            - ((np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 24]) * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + (0.+torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).cos()
            torso_lengths_angles_v[:, 45] = torso_lengths_angles_v[:, 39] - (0.+torso_lengths_angles_v[:, 20]) \
                                            + (-(0.+torso_lengths_angles_v[:, 24]) * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + ((np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 24]) * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + (0.+torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).sin()

            # left elbow in vectorized form
            torso_lengths_angles_v[:, 46] = torso_lengths_angles_v[:, 37] + (0.+torso_lengths_angles_v[:, 23]) \
                                            + (-(np.pi + torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 25]) * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos())
            torso_lengths_angles_v[:, 47] = torso_lengths_angles_v[:, 38] \
                                            + (-(0.+torso_lengths_angles_v[:, 25]) * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + ((np.pi + torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 25]) * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + (0.+torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).cos()
            torso_lengths_angles_v[:, 48] = torso_lengths_angles_v[:, 39] - (0.+torso_lengths_angles_v[:, 20]) \
                                            + (-(0.+torso_lengths_angles_v[:, 25]) * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            - ((np.pi + torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 25]) * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + (0.+torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).sin()

            # right hand in vectorized form
            torso_lengths_angles_v[:, 49] = torso_lengths_angles_v[:, 37] - (0.+torso_lengths_angles_v[:, 22]) \
                                            + ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  (0.+torso_lengths_angles_v[:, 26]) - (0.+torso_lengths_angles_v[:, 25])) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *  (0.+torso_lengths_angles_v[:, 26])) \
                                            + ((1.8 -torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 26])
            torso_lengths_angles_v[:, 50] = torso_lengths_angles_v[:, 38] \
                                            + (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  (0.+torso_lengths_angles_v[:, 26]) - (0.+torso_lengths_angles_v[:, 25])) + ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *(0.+torso_lengths_angles_v[:, 26])) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            - (((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 26]) - (0.+torso_lengths_angles_v[:, 25])) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 26])) - ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 26])) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + (0.+torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).cos()
            torso_lengths_angles_v[:, 51] = torso_lengths_angles_v[:, 39] - (0.+torso_lengths_angles_v[:, 20])\
                                            + (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  (0.+torso_lengths_angles_v[:, 26]) - (0.+torso_lengths_angles_v[:, 25])) + ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *(0.+torso_lengths_angles_v[:, 26])) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + (((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 26]) - (0.+torso_lengths_angles_v[:, 25])) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 26])) - ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 26])) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + (0.+torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).sin()

            # left hand in vectorized form
            torso_lengths_angles_v[:, 52] = torso_lengths_angles_v[:, 37] + (0.+torso_lengths_angles_v[:, 23]) \
                                            + ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() *  (0.+torso_lengths_angles_v[:, 27]) - (0.+torso_lengths_angles_v[:, 25])) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *  (0.+torso_lengths_angles_v[:, 27])) \
                                            + ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *  (0.+torso_lengths_angles_v[:, 27])
            torso_lengths_angles_v[:, 53] = torso_lengths_angles_v[:, 38] \
                                            + (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 27]) - (0.+torso_lengths_angles_v[:, 25])) + ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 27])) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            - (((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 27]) - (0.+torso_lengths_angles_v[:, 25])) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 27])) - ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 27])) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + (0.+torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).cos()
            torso_lengths_angles_v[:, 54] = torso_lengths_angles_v[:, 39] - (0.+torso_lengths_angles_v[:, 20])\
                                            + (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 27]) - (0.+torso_lengths_angles_v[:, 25])) + ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 27])) * (0. + torso_lengths_angles_v[:, 18]).sin() \
                                            + (((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 27]) - (0.+torso_lengths_angles_v[:, 25])) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 27])) - ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 27])) * (0. + torso_lengths_angles_v[:, 18]).cos() \
                                            + (0.+torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).sin()


            # right knee in vectorized form
            torso_lengths_angles_v[:, 55] = torso_lengths_angles_v[:, 37] - (0.+torso_lengths_angles_v[:, 31]) \
                                            + ((torso_lengths_angles_v[:, 14] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 33]) * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).cos())
            torso_lengths_angles_v[:, 56] = torso_lengths_angles_v[:, 38] \
                                            + (-(0.+torso_lengths_angles_v[:, 33]) * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            - ((torso_lengths_angles_v[:, 14] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 33]) * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - (0.+torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).cos()
            torso_lengths_angles_v[:, 57] = torso_lengths_angles_v[:, 39] - (0.+torso_lengths_angles_v[:, 29]) \
                                            + (-(0.+torso_lengths_angles_v[:, 33]) * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            + ((torso_lengths_angles_v[:, 14] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 33]) * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + (0.+torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).sin()

            # left knee in vectorized form
            torso_lengths_angles_v[:, 58] = torso_lengths_angles_v[:, 37] + (0.+torso_lengths_angles_v[:, 32]) \
                                            + (-((-1.8 - torso_lengths_angles_v[:, 15]) * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 34]) * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).cos())
            torso_lengths_angles_v[:, 59] = torso_lengths_angles_v[:, 38] \
                                            + (-(0.+torso_lengths_angles_v[:, 34]) * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + (((-1.8 - torso_lengths_angles_v[:, 15]) * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 34]) * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - (0.+torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).cos()
            torso_lengths_angles_v[:, 60] = torso_lengths_angles_v[:, 39] - (0.+torso_lengths_angles_v[:, 29]) \
                                            + (-(0.+torso_lengths_angles_v[:, 34]) * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - (((-1.8 - torso_lengths_angles_v[:, 15]) * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 34]) * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + (0.+torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).sin()

            # right ankle in vectorized form
            torso_lengths_angles_v[:, 61] = torso_lengths_angles_v[:, 37] - (0.+torso_lengths_angles_v[:, 31]) \
                                            - (-(torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).cos() * (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *  (0.+torso_lengths_angles_v[:, 35]) -  (0.+torso_lengths_angles_v[:, 33])) - ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() *  (0.+torso_lengths_angles_v[:, 35])) \
                                            + (-(torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 35])
            torso_lengths_angles_v[:, 62] = torso_lengths_angles_v[:, 38] \
                                            + (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *  (0.+torso_lengths_angles_v[:, 35]) - (0.+torso_lengths_angles_v[:, 33])) + ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() *(0.+torso_lengths_angles_v[:, 35])) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + (((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 35]) -  (0.+torso_lengths_angles_v[:, 33])) - ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 35])) - ((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 35])) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - (0.+torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).cos()
            torso_lengths_angles_v[:, 63] = torso_lengths_angles_v[:, 39] - (0.+torso_lengths_angles_v[:, 29]) \
                                            + (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *  (0.+torso_lengths_angles_v[:, 35]) - (0.+torso_lengths_angles_v[:, 33])) + ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() *(0.+torso_lengths_angles_v[:, 35])) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - (((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 35]) -  (0.+torso_lengths_angles_v[:, 33])) - ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 35])) - ((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 35])) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + (0.+torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).sin()

            # left ankle in vectorized form, need to fix third line of [:,65]
            torso_lengths_angles_v[:, 64] = torso_lengths_angles_v[:, 37] + (0.+torso_lengths_angles_v[:, 32]) \
                                            + (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).cos() * (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  (0.+torso_lengths_angles_v[:, 36]) - (0.+torso_lengths_angles_v[:, 34])) - ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() *  (0.+torso_lengths_angles_v[:, 36])) \
                                            + (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).sin() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() *  (0.+torso_lengths_angles_v[:, 36])
            torso_lengths_angles_v[:, 65] = torso_lengths_angles_v[:, 38] \
                                            + (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 36]) - (0.+torso_lengths_angles_v[:, 34])) + ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 36])) * (0. + torso_lengths_angles_v[:, 19]).cos()\
                                            - ((((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  (0.+torso_lengths_angles_v[:, 36]) - (0.+torso_lengths_angles_v[:, 34])) - ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 36])) - (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).sin() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 36])) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            - (0.+torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).cos()
            torso_lengths_angles_v[:, 66] = torso_lengths_angles_v[:, 39] - (0.+torso_lengths_angles_v[:, 29]) \
                                            + (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() * (0.+torso_lengths_angles_v[:, 36]) - (0.+torso_lengths_angles_v[:, 34])) + ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 36])) * (0. + torso_lengths_angles_v[:, 19]).sin() \
                                            + ((((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  (0.+torso_lengths_angles_v[:, 36]) - (0.+torso_lengths_angles_v[:, 34])) - ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 36])) - (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).sin() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * (0.+torso_lengths_angles_v[:, 36])) * (0. + torso_lengths_angles_v[:, 19]).cos() \
                                            + (0.+torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).sin()








            if forward_only == True:
                #let's get the neck, shoulders, and glutes pseudotargets
                pseudotargets = Variable(torch.Tensor(np.zeros((images.shape[0], 15))))

                #get the neck in vectorized form
                pseudotargets[:, 0] = torso_lengths_angles_v[:, 37]
                pseudotargets[:, 1] = torso_lengths_angles_v[:, 38] + (0. + torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).cos()
                pseudotargets[:, 2] = torso_lengths_angles_v[:, 39] - (0. + torso_lengths_angles_v[:, 20]) + (0. + torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).sin()

                #get the right shoulder in vectorized form
                pseudotargets[:, 3] = torso_lengths_angles_v[:, 37] - (0. + torso_lengths_angles_v[:, 22])
                pseudotargets[:, 4] = torso_lengths_angles_v[:, 38] + (0. + torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).cos()
                pseudotargets[:, 5] = torso_lengths_angles_v[:, 39] - (0. + torso_lengths_angles_v[:, 20]) + (0. + torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).sin()

                #print \the left shoulder in vectorized form
                pseudotargets[:, 6] = torso_lengths_angles_v[:, 37] + (0. + torso_lengths_angles_v[:, 23])
                pseudotargets[:, 7] = torso_lengths_angles_v[:, 38] + (0. + torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).cos()
                pseudotargets[:, 8] = torso_lengths_angles_v[:, 39] - (0. + torso_lengths_angles_v[:, 20]) + (0. + torso_lengths_angles_v[:, 21]) * (0. + torso_lengths_angles_v[:, 18]).sin()

                #get the right glute in vectorized form
                pseudotargets[:, 9] = torso_lengths_angles_v[:, 37] - (0. + torso_lengths_angles_v[:, 31])
                pseudotargets[:, 10] = torso_lengths_angles_v[:, 38] - (0. + torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).cos()
                pseudotargets[:, 11] = torso_lengths_angles_v[:, 39] - (0. + torso_lengths_angles_v[:, 29]) + (0. + torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).sin()

                #print \the left glute in vectorized form
                pseudotargets[:, 12] = torso_lengths_angles_v[:, 37] + (0. + torso_lengths_angles_v[:, 32])
                pseudotargets[:, 13] = torso_lengths_angles_v[:, 38] - (0. + torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).cos()
                pseudotargets[:, 14] = torso_lengths_angles_v[:, 39] - (0. + torso_lengths_angles_v[:, 29]) + (0. + torso_lengths_angles_v[:, 30]) * (0. + torso_lengths_angles_v[:, 19]).sin()

                pseudotargets = pseudotargets.data.numpy() * 1000

                print torso_lengths_angles_v[:,31:33]

            # angles = torso_lengths_angles_v[:, 0:20].data.numpy()*100
            angles = torso_lengths_angles[:, 0:20]*100
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = F.pad(torso_lengths_angles_v, (-20, 0, 0, 0)) #cut off all the angles
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)

            #print 'blah'


        return torso_lengths_angles_v, angles, pseudotargets

