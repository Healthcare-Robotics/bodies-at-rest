#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import matplotlib.gridspec as gridspec
import math



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

#ROS
import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf


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
NUMOFTAXELS_X = 74#73 #taxels
NUMOFTAXELS_Y = 27#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)




class VisualizationLib():

    def print_error_train(self, target, score, output_size, loss_vector_type = None, data = None, printerror = True):




        error = (score - target)

        error = np.reshape(error, (error.shape[0], output_size[0], output_size[1]))

        error_norm = np.expand_dims(np.linalg.norm(error, axis = 2),2)
        error = np.concatenate((error, error_norm), axis = 2)

        error_avg = np.mean(error, axis=0) / 10 #convert from mm to cm

       # for i in error_avg[:, 3]*10:
       #     print i

        error_avg = np.reshape(error_avg, (output_size[0], output_size[1]+1))
        error_avg_print = np.reshape(np.array(["%.2f" % w for w in error_avg.reshape(error_avg.size)]),
                                     (output_size[0], output_size[1] + 1))


        error_avg_print = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ',
                                                        'Pelvis ', 'L Hip  ', 'R Hip  ', 'Spine 1', 'L Knee ', 'R Knee ',
                                                        'Spine 2', 'L Ankle', 'R Ankle', 'Spine 3', 'L Foot ', 'R Foot ',
                                                        'Neck   ', 'L Sh.in', 'R Sh.in', 'Head   ', 'L Sh.ou', 'R Sh.ou',
                                                        'L Elbow', 'R Elbow', 'L Wrist', 'R Wrist', 'L Hand ', 'R Hand ']], np.transpose(
            np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg_print))))))
        if printerror == True:
            print data, error_avg_print


        error_std = np.std(error, axis=0) / 10

        #for i in error_std[:, 3]*10:
        #    print i

        error_std = np.reshape(error_std, (output_size[0], output_size[1] + 1))
        error_std_print = np.reshape(np.array(["%.2f" % w for w in error_std.reshape(error_std.size)]),
                                     (output_size[0], output_size[1] + 1))

        error_std_print = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ',
                                                        'Pelvis ', 'L Hip  ', 'R Hip  ', 'Spine 1', 'L Knee ', 'R Knee ',
                                                        'Spine 2', 'L Ankle', 'R Ankle', 'Spine 3', 'L Foot ', 'R Foot ',
                                                        'Neck   ', 'L Sh.in', 'R Sh.in', 'Head   ', 'L Sh.ou', 'R Sh.ou',
                                                        'L Elbow', 'R Elbow', 'L Wrist', 'R Wrist', 'L Hand ', 'R Hand ']], np.transpose(
                np.concatenate(([['', '', '', ''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std_print))))))
        #if printerror == True:
        #    print data, error_std_print
        error_norm = np.squeeze(error_norm, axis = 2)

        #return error_avg[:,3], error_std[:,3]
        return error_norm, error_avg[:,3], error_std[:,3]

    def print_error_val(self, target, score, output_size, loss_vector_type = None, data = None, printerror = True):

        if target.shape[1] == 72:
            target = target.reshape(-1, 24, 3)
            target = np.stack((target[:, 15, :],
                               target[:, 3, :],
                               target[:, 19, :],
                               target[:, 18, :],
                               target[:, 21, :],
                               target[:, 20, :],
                               target[:, 5, :],
                               target[:, 4, :],
                               target[:, 8, :],
                               target[:, 7, :],), axis = 1)

            score = score.reshape(-1, 24, 3)
            score = np.stack((score[:, 15, :],
                               score[:, 3, :],
                               score[:, 19, :],
                               score[:, 18, :],
                               score[:, 21, :],
                               score[:, 20, :],
                               score[:, 5, :],
                               score[:, 4, :],
                               score[:, 8, :],
                               score[:, 7, :],), axis = 1)


        error = (score - target)

        #print error.shape
        error = np.reshape(error, (error.shape[0], output_size[0], output_size[1]))

        #print error.shape

        error_norm = np.expand_dims(np.linalg.norm(error, axis = 2),2)
        error = np.concatenate((error, error_norm), axis = 2)

        error_avg = np.mean(error, axis=0) / 10

        #for i in error_avg[:, 3]*10:
        #    print i

        error_avg = np.reshape(error_avg, (output_size[0], output_size[1]+1))
        error_avg_print = np.reshape(np.array(["%.2f" % w for w in error_avg.reshape(error_avg.size)]),
                                     (output_size[0], output_size[1] + 1))


        error_avg_print = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', 'Head   ',
                                                   'Torso  ', 'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ',
                                                   'R Knee ', 'L Knee ', 'R Foot ', 'L Foot ']], np.transpose(
            np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg_print))))))
        if printerror == True:
            print data, error_avg_print


        error_std = np.std(error, axis=0) / 10

        #for i in error_std[:, 3]*10:
        #    print i

        error_std = np.reshape(error_std, (output_size[0], output_size[1] + 1))
        error_std_print = np.reshape(np.array(["%.2f" % w for w in error_std.reshape(error_std.size)]),
                                     (output_size[0], output_size[1] + 1))

        error_std_print = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ', 'Head   ', 'Torso  ',
                              'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ', 'R Knee ', 'L Knee ',
                              'R Foot ', 'L Foot ']], np.transpose(
                np.concatenate(([['', '', '', ''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std_print))))))
        #if printerror == True:
        #    print data, error_std_print
        error_norm = np.squeeze(error_norm, axis = 2)

        #return error_avg[:,3], error_std[:,3]
        return error_norm, error_avg[:,3], error_std[:,3]

    def print_error_iros2018(self, target, score, output_size, loss_vector_type = None, data = None, printerror = True):


        if target.shape[1] == 72:
            target = target.reshape(-1, 24, 3)
            target = np.stack((target[:, 15, :],
                               target[:, 3, :],
                               target[:, 19, :],
                               target[:, 18, :],
                               target[:, 21, :],
                               target[:, 20, :],
                               target[:, 5, :],
                               target[:, 4, :],
                               target[:, 8, :],
                               target[:, 7, :],), axis = 1)

            score = score.reshape(-1, 24, 3)
            score = np.stack((score[:, 15, :],
                               score[:, 3, :],
                               score[:, 19, :],
                               score[:, 18, :],
                               score[:, 21, :],
                               score[:, 20, :],
                               score[:, 5, :],
                               score[:, 4, :],
                               score[:, 8, :],
                               score[:, 7, :],), axis = 1)


        error = (score - target)

        print error.shape
        error = np.reshape(error, (error.shape[0], output_size[0], output_size[1]))

        print error.shape

        error_norm = np.expand_dims(np.linalg.norm(error, axis = 2),2)
        error = np.concatenate((error, error_norm), axis = 2)

        error_avg = np.mean(error, axis=0) / 10

        for i in error_avg[:, 3]*10:
            print i

        error_avg = np.reshape(error_avg, (output_size[0], output_size[1]+1))
        error_avg_print = np.reshape(np.array(["%.2f" % w for w in error_avg.reshape(error_avg.size)]),
                                     (output_size[0], output_size[1] + 1))


        error_avg_print = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', 'Head   ',
                                                   'Torso  ', 'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ',
                                                   'R Knee ', 'L Knee ', 'R Foot ', 'L Foot ']], np.transpose(
            np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg_print))))))
        if printerror == True:
            print data, error_avg_print


        error_std = np.std(error, axis=0) / 10

        for i in error_std[:, 3]*10:
            print i

        error_std = np.reshape(error_std, (output_size[0], output_size[1] + 1))
        error_std_print = np.reshape(np.array(["%.2f" % w for w in error_std.reshape(error_std.size)]),
                                     (output_size[0], output_size[1] + 1))

        error_std_print = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ', 'Head   ', 'Torso  ',
                              'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ', 'R Knee ', 'L Knee ',
                              'R Foot ', 'L Foot ']], np.transpose(
                np.concatenate(([['', '', '', ''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std_print))))))
        if printerror == True:
            print data, error_std_print
        error_norm = np.squeeze(error_norm, axis = 2)

        #return error_avg[:,3], error_std[:,3]
        return error_norm, error_avg[:,3], error_std[:,3]


    def visualize_error_from_distance(self, bed_distance, error_norm):
        plt.close()
        fig = plt.figure()
        ax = []
        for joint in range(0, bed_distance.shape[1]):
            ax.append(joint)
            if bed_distance.shape[1] <= 5:
                ax[joint] = fig.add_subplot(1, bed_distance.shape[1], joint + 1)
            else:
                #print math.ceil(bed_distance.shape[1]/2.)
                ax[joint] = fig.add_subplot(2, math.ceil(bed_distance.shape[1]/2.), joint + 1)
            ax[joint].set_xlim([0, 0.7])
            ax[joint].set_ylim([0, 1])
            ax[joint].set_title('Joint ' + str(joint) + ' error')
            ax[joint].plot(bed_distance[:, joint], error_norm[:, joint], 'r.')
        plt.show()


    def visualize_pressure_map(self, p_map, targets_raw=None, scores_raw = None,
                                p_map_val = None, targets_val = None, scores_val = None,
                                p_map_ext = None, targets_ext = None, scores_ext = None,
                                p_map_ext2 = None, targets_ext2 = None, scores_ext2 = None,
                                block = False, title = ' '):

        p_map_mult = 1.
        p_map_val_mult = 1.
        p_map_ext_mult = 1.
        p_map_ext2_mult = 1.

        try:
            p_map = p_map[0,:,:] #select the original image matrix from the intermediate amplifier matrix and the height matrix
            if p_map.shape[0] == 128: p_map_mult = 2.
        except:
            if p_map.shape[0] == 128: p_map_mult = 2.

        plt.close()
        plt.pause(0.0001)


        # set options
        num_subplots = 1
        if p_map_val is not None:
            try:
                p_map_val = p_map_val[0, :, :]  # select the original image matrix from the intermediate amplifier matrix and the height matrix
                if p_map_val.shape[0] == 128: p_map_val_mult = 2.
            except:
                if p_map_val.shape[0] == 128: p_map_val_mult = 2.
            num_subplots += 1
        if p_map_ext is not None:
            try:
                p_map_ext = p_map_ext[0, :, :]  # select the original image matrix from the intermediate amplifier matrix and the height matrix
                if p_map_ext.shape[0] == 128: p_map_ext_mult = 2.
            except:
                if p_map_ext.shape[0] == 128: p_map_ext_mult = 2.
            num_subplots += 1
        if p_map_ext2 is not None:
            try:
                p_map_ext2 = p_map_ext2[0, :, :]  # select the original image matrix from the intermediate amplifier matrix and the height matrix
                if p_map_ext2.shape[0] == 128: p_map_ext2_mult = 2.
            except:
                if p_map_ext2.shape[0] == 128: p_map_ext2_mult = 2.
            num_subplots += 1


        fig = plt.figure(figsize = (3*num_subplots, 5))
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        #mngr.window.setGeometry(50, 100, 840, 705)

        plt.pause(0.0001)


        ax1 = fig.add_subplot(1, num_subplots, 1)
        ax1.set_xlim([-10.0*p_map_mult, 37.0*p_map_mult])
        ax1.set_ylim([74.0*p_map_mult, -10.0*p_map_mult])
        ax1.set_facecolor('cyan')
        ax1.imshow(p_map, interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax1.set_title('Training Sample \n Pressure Mat, \n Targets and Estimates')
        ax2 = None
        ax3 = None
        ax4 = None

        if p_map_val is not None:
            ax2 = fig.add_subplot(1, num_subplots, 2)
            ax2.set_xlim([-10.0*p_map_val_mult, 37.0*p_map_val_mult])
            ax2.set_ylim([74.0*p_map_val_mult, -10.0*p_map_val_mult])
            ax2.set_facecolor('cyan')
            ax2.imshow(p_map_val, interpolation='nearest', cmap=
            plt.cm.jet, origin='upper', vmin=0, vmax=100)
            ax2.set_title('Training Sample \n Est. Input \n Mesh Penetration')

        if p_map_ext is not None:
            ax3 = fig.add_subplot(1, num_subplots, 3)
            ax3.set_xlim([-10.0*p_map_ext_mult, 37.0*p_map_ext_mult])
            ax3.set_ylim([74.0*p_map_ext_mult, -10.0*p_map_ext_mult])
            #ax3.set_facecolor('cyan')
            ax3.imshow(p_map_ext, interpolation='nearest', cmap=
            plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax3.set_title('Training Sample \n Ground Truth \n Mesh Penetration')

        if p_map_ext2 is not None:
            ax4 = fig.add_subplot(1, num_subplots, 4)
            ax4.set_xlim([-10.0*p_map_ext2_mult, 37.0*p_map_ext2_mult])
            ax4.set_ylim([74.0*p_map_ext2_mult, -10.0*p_map_ext2_mult])
            #ax4.set_facecolor('cyan')
            ax4.imshow(p_map_ext2, interpolation='nearest', cmap=
            plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax4.set_title('Training Sample \n Est. Current \n Mesh Penetration')
            #ax4.set_title('Validation Sample \n Pressure Mat, \n Targets and Estimates')


        # Visualize targets of training set
        self.plot_joint_markers(targets_raw, p_map_mult, ax1, 'green')

        #Visualize estimated from training set
        self.plot_joint_markers(scores_raw, p_map_mult, ax1, 'yellow')

        # Visualize targets of validation set
        self.plot_joint_markers(targets_val, p_map_val_mult, ax2, 'green')

        # Visualize estimated from val set
        self.plot_joint_markers(scores_val, p_map_val_mult, ax2, 'yellow')

        # Visualize targets of extra set
        self.plot_joint_markers(targets_ext, p_map_ext_mult, ax3, 'green')

        # Visualize estimated from extra set
        self.plot_joint_markers(scores_ext, p_map_ext_mult, ax3, 'yellow')

        # Visualize targets of extra set
        self.plot_joint_markers(targets_ext2, p_map_ext2_mult, ax4, 'green')

        # Visualize estimated from extra set
        self.plot_joint_markers(scores_ext2, p_map_ext2_mult, ax4, 'yellow')

        axkeep = plt.axes([0.01, 0.05, 0.15, 0.075])
        axdisc = plt.axes([0.01, 0.15, 0.15, 0.075])
        self.skip_image = False
        bdisc = Button(axdisc, 'Skip Image')
        bdisc.on_clicked(self.skip)
        bkeep = Button(axkeep, 'Continue')
        bkeep.on_clicked(self.cont)

        plt.show(block=block)

        return self.skip_image


    def plot_joint_markers(self, markers, p_map_mult, ax, color):
        if markers is not None:
            if len(np.shape(markers)) == 1:
                markers = np.reshape(markers, (len(markers) / 3, 3))
            target_coord = np.array(markers[:, :2]) / INTER_SENSOR_DISTANCE
            target_coord[:, 0] -= 10
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            target_coord*=p_map_mult
            ax.plot(target_coord[:, 0], target_coord[:, 1], marker = 'o', linestyle='None', markerfacecolor = color, markeredgecolor='black', ms=8)
        plt.pause(0.0001)


    def skip(self, event):
        plt.close()
        self.skip_image = True

    def cont(self, event):
        plt.close()
        self.skip_image = False




    def rviz_publish_input(self, image, angle):
        mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)

        markerArray = MarkerArray()
        for j in range(image.shape[0]):
            for i in range(image.shape[1]):
                imagePublisher = rospy.Publisher("/pressure_image", MarkerArray)

                marker = Marker()
                marker.header.frame_id = "map"
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.a = 1.0
                if image[j,i] > 60:
                    marker.color.r = 1.
                    marker.color.b = (100 - image[j, i])*.9 / 60.
                else:
                    marker.color.r = image[j, i] / 40.
                    marker.color.b = 1.
                marker.color.g = (70-np.abs(image[j,i]-50))/100.

                marker.pose.orientation.w = 1.0

                marker.pose.position.x = i*0.0286
                if j > 23:
                    marker.pose.position.y = (64-j)*0.0286 - 0.0286*3*np.sin(np.deg2rad(angle))
                    marker.pose.position.z = 0.0#-0.1
                    #print marker.pose.position.x, 'x'
                else:

                    marker.pose.position.y = (41) * 0.0286 + (23 - j) * 0.0286 * np.cos(np.deg2rad(angle)) - (0.0286*3*np.sin(np.deg2rad(angle)))*0.85
                    marker.pose.position.z = ((23-j)*0.0286*np.sin(np.deg2rad(angle)))*0.85 #-0.1
                    #print j, marker.pose.position.z, marker.pose.position.y, 'head'

                # We add the new marker to the MarkerArray, removing the oldest
                # marker from it when necessary
                #if (self.count > 100):
                 #   markerArray.markers.pop(0)

                markerArray.markers.append(marker)

                #print self.count

                # Renumber the marker IDs
                id = 0
                for m in markerArray.markers:
                    m.id = id
                    id += 1
        imagePublisher.publish(markerArray)


    def rviz_publish_output(self, targets, scores = None, scores_std = None):

        if targets is not None:
            TargetArray = MarkerArray()
            for joint in range(0, targets.shape[0]):
                targetPublisher = rospy.Publisher("/targets", MarkerArray)
                Tmarker = Marker()
                Tmarker.header.frame_id = "map"
                Tmarker.type = Tmarker.SPHERE
                Tmarker.action = Tmarker.ADD
                Tmarker.scale.x = 0.07
                Tmarker.scale.y = 0.07
                Tmarker.scale.z = 0.07
                Tmarker.color.a = 1.0
                Tmarker.color.r = 0.0
                Tmarker.color.g = 0.69
                Tmarker.color.b = 0.0
                Tmarker.pose.orientation.w = 1.0
                Tmarker.pose.position.x = targets[joint, 0] - INTER_SENSOR_DISTANCE*10
                Tmarker.pose.position.y = targets[joint, 1] - INTER_SENSOR_DISTANCE*10
                Tmarker.pose.position.z = targets[joint, 2]
                TargetArray.markers.append(Tmarker)
                tid = 0
                for m in TargetArray.markers:
                    m.id = tid
                    tid += 1
            #print TargetArray
            targetPublisher.publish(TargetArray)

        if scores is not None:
            ScoresArray = MarkerArray()
            for joint in range(0, scores.shape[0]):
                scoresPublisher = rospy.Publisher("/scores", MarkerArray)
                Smarker = Marker()
                Smarker.header.frame_id = "map"
                Smarker.type = Smarker.SPHERE
                Smarker.action = Smarker.ADD
                Smarker.scale.x = 0.06
                Smarker.scale.y = 0.06
                Smarker.scale.z = 0.06
                Smarker.color.a = 1.0
                if scores_std is not None:
                    #print scores_std[joint], 'std of joint ', joint
                    #std of 3 is really uncertain
                    Smarker.color.r = 1.0
                    Smarker.color.g = 1.0 - scores_std[joint]/0.05
                    Smarker.color.b = scores_std[joint]/0.05

                else:
                    if joint == 1:
                        Smarker.color.r = 1.0
                        Smarker.color.g = 1.0
                    else:
                        Smarker.color.r = 1.0
                        Smarker.color.g = 1.0
                    Smarker.color.b = 0.0

                Smarker.pose.orientation.w = 1.0
                Smarker.pose.position.x = scores[joint, 0] - INTER_SENSOR_DISTANCE*10
                Smarker.pose.position.y = scores[joint, 1] - INTER_SENSOR_DISTANCE*10
                Smarker.pose.position.z = scores[joint, 2]
                ScoresArray.markers.append(Smarker)
                sid = 0
                for m in ScoresArray.markers:
                    m.id = sid
                    sid += 1
            scoresPublisher.publish(ScoresArray)


    def rviz_publish_output_mesh(self, targets, scores = None):

        if targets is not None:
            TargetArray = MarkerArray()
            for joint in range(0, targets.shape[0]):
                targetPublisher = rospy.Publisher("/mesh", MarkerArray)
                Tmarker = Marker()
                Tmarker.header.frame_id = "map"
                Tmarker.type = Tmarker.SPHERE
                Tmarker.action = Tmarker.ADD
                Tmarker.scale.x = 0.01
                Tmarker.scale.y = 0.01
                Tmarker.scale.z = 0.01
                Tmarker.color.a = 0.5
                Tmarker.color.r = 0.8
                Tmarker.color.g = 0.8
                Tmarker.color.b = 0.8
                Tmarker.pose.orientation.w = 1.0
                Tmarker.pose.position.x = targets[joint, 0] - INTER_SENSOR_DISTANCE*10
                Tmarker.pose.position.y = targets[joint, 1] - INTER_SENSOR_DISTANCE*10
                Tmarker.pose.position.z = targets[joint, 2]
                TargetArray.markers.append(Tmarker)
                tid = 0
                for m in TargetArray.markers:
                    m.id = tid
                    tid += 1
            #print TargetArray
            targetPublisher.publish(TargetArray)

        if scores is not None:
            ScoreArray = MarkerArray()
            for joint in range(0, scores.shape[0]):
                scorePublisher = rospy.Publisher("/mesh", MarkerArray)
                Tmarker = Marker()
                Tmarker.header.frame_id = "map"
                Tmarker.type = Tmarker.SPHERE
                Tmarker.action = Tmarker.ADD
                Tmarker.scale.x = 0.07
                Tmarker.scale.y = 0.07
                Tmarker.scale.z = 0.07
                Tmarker.color.a = 1.0
                Tmarker.color.r = 0.8
                Tmarker.color.g = 0.8
                Tmarker.color.b = 0.8
                Tmarker.pose.orientation.w = 1.0
                Tmarker.pose.position.x = scores[joint, 0] - INTER_SENSOR_DISTANCE*10
                Tmarker.pose.position.y = scores[joint, 1] - INTER_SENSOR_DISTANCE*10
                Tmarker.pose.position.z = scores[joint, 2]
                ScoreArray.markers.append(Tmarker)
                tid = 0
                for m in ScoreArray.markers:
                    m.id = tid
                    tid += 1
            #print scoreArray
            scorePublisher.publish(ScoreArray)





    def rviz_publish_output_limbs(self, targets, scores, pseudotargets = None, LimbArray = None, count = 0):
        print 'publishing for kinematics model!'
        #if LimbArray == None or count <= 2:
        LimbArray = MarkerArray()

        limbs = {}
        limbs['right_forearm'] = [scores[2,0], scores[2,1], scores[2,2], scores[4,0], scores[4,1], scores[4,2]]
        limbs['left_forearm'] = [scores[3,0], scores[3,1], scores[3,2], scores[5,0], scores[5,1], scores[5,2]]
        limbs['right_upperarm'] = [pseudotargets[1,0], pseudotargets[1,1], pseudotargets[1,2], scores[2,0], scores[2,1], scores[2,2]]
        limbs['left_upperarm'] = [pseudotargets[2,0], pseudotargets[2,1], pseudotargets[2,2], scores[3,0], scores[3,1], scores[3,2]]

        limbs['right_calf'] = [scores[6,0], scores[6,1], scores[6,2], scores[8,0], scores[8,1], scores[8,2]]
        limbs['left_calf'] = [scores[7,0], scores[7,1], scores[7,2], scores[9,0], scores[9,1], scores[9,2]]
        limbs['right_thigh'] = [pseudotargets[3,0], pseudotargets[3,1], pseudotargets[3,2], scores[6,0], scores[6,1], scores[6,2]]
        limbs['left_thigh'] = [pseudotargets[4,0], pseudotargets[4,1], pseudotargets[4,2], scores[7,0], scores[7,1], scores[7,2]]
        limbs['torso_drop'] = [scores[1,0], scores[1,1], scores[1,2], scores[1,0]+.0001, scores[1,1]+.0001, scores[1,2] - .12]
        limbs['torso_midhip'] = [scores[1,0]+.0001, scores[1,1]+.0001, scores[1,2] - .12, (pseudotargets[4,0]+pseudotargets[3,0])/2, pseudotargets[4,1], pseudotargets[4,2]]
        limbs['torso_neck'] = [scores[1,0]+.0001, scores[1,1]+.0001, scores[1,2] - .12, (pseudotargets[1,0]+pseudotargets[2,0])/2, pseudotargets[1,1], pseudotargets[1,2]]
        limbs['neck_head'] = [(pseudotargets[1,0]+pseudotargets[2,0])/2, pseudotargets[1,1], pseudotargets[1,2], scores[0,0], scores[0,1], scores[0,2]]

        limbs['shouldershoulder'] =  [pseudotargets[1,0]+.0001, pseudotargets[1,1]+.0001, pseudotargets[1,2]+.0001, pseudotargets[2,0], pseudotargets[2,1], pseudotargets[2,2]]
        limbs['hiphip'] =  [pseudotargets[3,0]+.0001, pseudotargets[3,1]+.0001, pseudotargets[3,2]+.0001, pseudotargets[4,0], pseudotargets[4,1], pseudotargets[4,2]]
        #limbs['left_upperarm'] = [scores[3,0], scores[3,1], scores[3,2], scores[5,0], scores[5,1], scores[5,2]]

        for limb in limbs:
            sx1 = limbs[limb][0]
            sy1 = limbs[limb][1]
            sz1 = limbs[limb][2]
            sx2 = limbs[limb][3]
            sy2 = limbs[limb][4]
            sz2 = limbs[limb][5]

            limbscorePublisher = rospy.Publisher("/limbscores", MarkerArray)
            Lmarker = Marker()
            Lmarker.header.frame_id = "autobed/base_link"
            Lmarker.type = Lmarker.CYLINDER
            Lmarker.action = Lmarker.ADD
            x_origin = np.array([1., 0., 0.])
            z_vector = np.array([(sx2-sx1), (sy2-sy1), (sz2-sz1)])
            z_mag = np.linalg.norm(z_vector)
            z_vector = z_vector / z_mag

            y_orth = np.cross(z_vector, x_origin)
            y_orth = y_orth / np.linalg.norm(y_orth)

            x_orth = np.cross(y_orth, z_vector)
            x_orth = x_orth / np.linalg.norm(x_orth)

            ROT_mat = np.matrix(np.eye(4))
            ROT_mat[0:3, 0] = np.copy(np.reshape(x_orth, [3,1]))
            ROT_mat[0:3, 1] = np.copy(np.reshape(y_orth, [3,1]))
            ROT_mat[0:3, 2] = np.copy(np.reshape(z_vector, [3,1]))
            Lmarker.scale.z = z_mag

            if count <= 0:
                Lmarker.color.a = 1.0
                Lmarker.scale.x = 0.025
                Lmarker.scale.y = 0.025
            else:
                Lmarker.color.a = 0.4
                Lmarker.scale.x = 0.015
                Lmarker.scale.y = 0.015

            Lmarker.color.r = 1.0
            Lmarker.color.g = 1.0
            Lmarker.color.b = 0.0
            Lmarker.pose.orientation.x = tf.transformations.quaternion_from_matrix(ROT_mat)[0]
            Lmarker.pose.orientation.y = tf.transformations.quaternion_from_matrix(ROT_mat)[1]
            Lmarker.pose.orientation.z = tf.transformations.quaternion_from_matrix(ROT_mat)[2]
            Lmarker.pose.orientation.w = tf.transformations.quaternion_from_matrix(ROT_mat)[3]

            Lmarker.pose.position.x = (sx1+sx2)/2
            Lmarker.pose.position.y = (sy1+sy2)/2
            Lmarker.pose.position.z = (sz1+sz2)/2
            LimbArray.markers.append(Lmarker)
            lid = 0
            for m in LimbArray.markers:
                m.id = lid
                lid += 1


        limbscorePublisher.publish(LimbArray)

        return LimbArray
    def rviz_publish_output_limbs_direct(self, targets, scores, LimbArray = None, count = 0):

        #if LimbArray == None or count <= 2:
        LimbArray = MarkerArray()

        limbs = {}
        limbs['right_forearm'] = [scores[2,0], scores[2,1], scores[2,2], scores[4,0], scores[4,1], scores[4,2]]
        limbs['left_forearm'] = [scores[3,0], scores[3,1], scores[3,2], scores[5,0], scores[5,1], scores[5,2]]

        limbs['right_calf'] = [scores[6,0], scores[6,1], scores[6,2], scores[8,0], scores[8,1], scores[8,2]]
        limbs['left_calf'] = [scores[7,0], scores[7,1], scores[7,2], scores[9,0], scores[9,1], scores[9,2]]

        for limb in limbs:
            sx1 = limbs[limb][0]
            sy1 = limbs[limb][1]
            sz1 = limbs[limb][2]
            sx2 = limbs[limb][3]
            sy2 = limbs[limb][4]
            sz2 = limbs[limb][5]

            limbscorePublisher = rospy.Publisher("/limbscoresdirect", MarkerArray)
            Lmarker = Marker()
            Lmarker.header.frame_id = "autobed/base_link"
            Lmarker.type = Lmarker.CYLINDER
            Lmarker.action = Lmarker.ADD
            x_origin = np.array([1., 0., 0.])
            z_vector = np.array([(sx2-sx1), (sy2-sy1), (sz2-sz1)])
            z_mag = np.linalg.norm(z_vector)
            z_vector = z_vector / z_mag

            y_orth = np.cross(z_vector, x_origin)
            y_orth = y_orth / np.linalg.norm(y_orth)

            x_orth = np.cross(y_orth, z_vector)
            x_orth = x_orth / np.linalg.norm(x_orth)

            ROT_mat = np.matrix(np.eye(4))
            ROT_mat[0:3, 0] = np.copy(np.reshape(x_orth, [3,1]))
            ROT_mat[0:3, 1] = np.copy(np.reshape(y_orth, [3,1]))
            ROT_mat[0:3, 2] = np.copy(np.reshape(z_vector, [3,1]))


            Lmarker.scale.z = z_mag

            if count <= 0:
                Lmarker.color.a = 1.0
                Lmarker.scale.x = 0.025
                Lmarker.scale.y = 0.025
            else:
                Lmarker.color.a = 0.5
                Lmarker.scale.x = 0.015
                Lmarker.scale.y = 0.015

            Lmarker.color.r = 1.0
            Lmarker.color.g = 1.0
            Lmarker.color.b = 0.0
            Lmarker.pose.orientation.x = tf.transformations.quaternion_from_matrix(ROT_mat)[0]
            Lmarker.pose.orientation.y = tf.transformations.quaternion_from_matrix(ROT_mat)[1]
            Lmarker.pose.orientation.z = tf.transformations.quaternion_from_matrix(ROT_mat)[2]
            Lmarker.pose.orientation.w = tf.transformations.quaternion_from_matrix(ROT_mat)[3]

            Lmarker.pose.position.x = (sx1+sx2)/2
            Lmarker.pose.position.y = (sy1+sy2)/2
            Lmarker.pose.position.z = (sz1+sz2)/2
            LimbArray.markers.append(Lmarker)
            lid = 0
            for m in LimbArray.markers:
                m.id = lid
                lid += 1


        limbscorePublisher.publish(LimbArray)

        return LimbArray


