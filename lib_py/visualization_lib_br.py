#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import matplotlib.gridspec as gridspec
import math
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import cPickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom

#ROS
#import rospy
#from visualization_msgs.msg import Marker
#from visualization_msgs.msg import MarkerArray
#import tf


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


    def visualize_pressure_map(self, pimage_in, cntct_in = None, sobel_in = None,
                                targets_raw=None, scores_net1 = None, scores_net2 = None,
                                pmap_recon_in = None, cntct_recon_in = None, hover_recon_in = None,
                                pmap_recon = None, cntct_recon = None, hover_recon = None,
                                pmap_recon_gt=None, cntct_recon_gt = None,
                                block = False, title = ' '):



        pimage_in_mult = 1.
        cntct_in_mult = 1.
        sobel_in_mult = 1.
        pmap_recon_in_mult = 1.
        cntct_recon_in_mult = 1.
        hover_recon_in_mult = 1.

        if pimage_in.shape[0] == 128: pimage_in_mult = 2.
        if cntct_in is not None:
            if cntct_in.shape[0] == 128: cntct_in_mult = 2.
        if sobel_in is not None:
            if sobel_in.shape[0] == 128: sobel_in_mult = 2.
        if pmap_recon_in is not None:
            if pmap_recon_in.shape[0] == 128: pmap_recon_in_mult = 2.
        if cntct_recon_in is not None:
            if cntct_recon_in.shape[0] == 128: cntct_recon_in_mult = 2.
        if hover_recon_in is not None:
            if hover_recon_in.shape[0] == 128: hover_recon_in_mult = 2.

        plt.close()
        plt.pause(0.0001)


        # set options
        num_subplots = 5
        if pmap_recon_in is not None:
            num_subplots += 3



        fig = plt.figure(tight_layout=True, figsize = (1.5*num_subplots*.8, 5*.8))
        gs = gridspec.GridSpec(2, num_subplots)


        plt.pause(0.0001)
        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')


        ax1 = fig.add_subplot(gs[:, 0:2])
        ax1.set_xlim([-10.0*pimage_in_mult, 37.0*pimage_in_mult])
        ax1.set_ylim([74.0*pimage_in_mult, -10.0*pimage_in_mult])
        ax1.set_facecolor('cyan')
        ax1.imshow(pimage_in, interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax1.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        ax1.set_title('Training Sample \n Pressure Image, \n Targets and Estimates')



        ax13 = fig.add_subplot(gs[0, 2])
        ax13.set_xlim([-pimage_in_mult, 27.0*pimage_in_mult])
        ax13.set_ylim([64.0*pimage_in_mult, -pimage_in_mult])
        ax13.imshow(pimage_in, interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax13.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        ax13.set_ylabel('INPUT')
        ax13.set_title(r'$\mathcal{P}$')

        ax14 = fig.add_subplot(gs[0, 3])
        ax14.set_xlim([-cntct_in_mult, 27.0 * cntct_in_mult])
        ax14.set_ylim([64.0 * cntct_in_mult, -cntct_in_mult])
        ax14.imshow(100 - cntct_in, interpolation='nearest', cmap=
        plt.cm.gray, origin='upper', vmin=0, vmax=100)
        ax14.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        ax14.set_title(r'$\mathcal{C}_I$')

        ax15 = fig.add_subplot(gs[0, 4])
        ax15.set_xlim([-sobel_in_mult, 27.0 * sobel_in_mult])
        ax15.set_ylim([64.0 * sobel_in_mult, -sobel_in_mult])
        ax15.imshow(sobel_in, interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax15.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
        ax15.set_title(r'$\mathcal{E}$')


        if pmap_recon_in is not None:
            ax16 = fig.add_subplot(gs[0, 5])
            ax16.set_xlim([-pmap_recon_in_mult, 27.0 * pmap_recon_in_mult])
            ax16.set_ylim([64.0 * pmap_recon_in_mult, -pmap_recon_in_mult])
            ax16.imshow(pmap_recon_in, interpolation='nearest', cmap=
            plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax16.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax16.set_title(r'$\hat{Q}^{-}_1$')

            ax17 = fig.add_subplot(gs[0, 6])
            ax17.set_xlim([-cntct_recon_in_mult, 27.0 * cntct_recon_in_mult])
            ax17.set_ylim([64.0 * cntct_recon_in_mult, -cntct_recon_in_mult])
            ax17.imshow(100 - cntct_recon_in*100, interpolation='nearest', cmap=
            plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax17.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax17.set_title(r'$\hat{C}_{O,1}$')

            ax18 = fig.add_subplot(gs[0, 7])
            ax18.set_xlim([-hover_recon_in_mult, 27.0 * hover_recon_in_mult])
            ax18.set_ylim([64.0 * hover_recon_in_mult, -hover_recon_in_mult])
            ax18.imshow(hover_recon_in, interpolation='nearest', cmap=
            plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax18.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, top=False)
            ax18.set_title(r'$\hat{Q}^{+}_1$')


        if pmap_recon is not None:
            q_out_subscr = str(1)
            if pmap_recon_in is not None:
                q_out_subscr = str(2)

            ax23 = fig.add_subplot(gs[1, 2])
            ax23.set_xlim([-1.0, 27.0])
            ax23.set_ylim([64.0, -1.0])
            ax23.imshow(pmap_recon, interpolation='nearest', cmap= plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax23.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax23.set_ylabel('OUTPUT')
            ax23.set_title(r'$\hat{Q}^{-}_'+q_out_subscr+'$')

            ax24 = fig.add_subplot(gs[1, 3])
            ax24.set_xlim([-1.0, 27.0])
            ax24.set_ylim([64.0, -1.0])
            ax24.imshow(100 - cntct_recon*100, interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax24.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax24.set_title(r'$\hat{C}_{O,'+q_out_subscr+'}$')

            ax25 = fig.add_subplot(gs[1, 4])
            ax25.set_xlim([-1.0, 27.0])
            ax25.set_ylim([64.0, -1.0])
            ax25.imshow(hover_recon, interpolation='nearest', cmap= plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax25.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax25.set_title(r'$\hat{Q}^{+}_'+q_out_subscr+'$')

        if pmap_recon_gt is not None:
            ax27 = fig.add_subplot(gs[1, 6])
            ax27.set_xlim([-1.0, 27.0])
            ax27.set_ylim([64.0, -1.0])
            ax27.imshow(pmap_recon_gt, interpolation='nearest', cmap= plt.cm.viridis, origin='upper', vmin=0, vmax=100)
            ax27.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax27.set_ylabel('GROUND TRUTH')
            ax27.set_title(r'$Q^{-}$')

            ax28 = fig.add_subplot(gs[1, 7])
            ax28.set_xlim([-1.0, 27.0])
            ax28.set_ylim([64.0, -1.0])
            ax28.imshow(100 - cntct_recon_gt*100, interpolation='nearest', cmap= plt.cm.gray, origin='upper', vmin=0, vmax=100)
            ax28.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom = False, top = False)
            ax28.set_title(r'$C_{O}$')


        # Visualize targets of training set
        self.plot_joint_markers(targets_raw, pimage_in_mult, ax1, 'green')

        #Visualize estimated from training set
        self.plot_joint_markers(scores_net1, pimage_in_mult, ax1, 'yellow')

        #fig.savefig('/home/henry/data/blah.png', dpi=400)
        plt.show(block=block)




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

