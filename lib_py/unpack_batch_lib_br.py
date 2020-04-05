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

import sys
sys.path.insert(0, '../lib_py')

from kinematics_lib_br import KinematicsLib
from preprocessing_lib_br import PreprocessingLib

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



# import hrl_lib.util as ut
import cPickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class UnpackBatchLib():


    def unpack_batch(self, batch, is_training, model, CTRL_PNL):

        INPUT_DICT = {}
        adj_ext_idx = 0
        # 0:72: positions.
        batch.append(batch[1][:, 72:82])  # betas
        batch.append(batch[1][:, 82:154])  # angles
        batch.append(batch[1][:, 154:157])  # root pos
        batch.append(batch[1][:, 157:159])  # gender switch
        batch.append(batch[1][:, 159])  # synth vs real switch
        batch.append(batch[1][:, 160:161])  # mass, kg
        batch.append(batch[1][:, 161:162])  # height, kg

        if CTRL_PNL['adjust_ang_from_est'] == True:
            adj_ext_idx += 3
            batch.append(batch[1][:, 162:172]) #betas est
            batch.append(batch[1][:, 172:244]) #angles est
            batch.append(batch[1][:, 244:247]) #root pos est
            if CTRL_PNL['full_body_rot'] == True:
                adj_ext_idx += 1
                batch.append(batch[1][:, 247:253]) #root atan2 est
                #print "appended root", batch[-1], batch[12]

            extra_smpl_angles = batch[10]
            extra_targets = batch[11]
        else:
            extra_smpl_angles = None
            extra_targets = None


        if CTRL_PNL['depth_map_labels'] == True:
            if CTRL_PNL['depth_map_labels_test'] == True or is_training == True:
                batch.append(batch[0][:, CTRL_PNL['num_input_channels_batch0'], : ,:]) #mesh depth matrix
                batch.append(batch[0][:, CTRL_PNL['num_input_channels_batch0']+1, : ,:]) #mesh contact matrix

                #cut off batch 0 so we don't have depth or contact on the input
                batch[0] = batch[0][:, 0:CTRL_PNL['num_input_channels_batch0'], :, :]

        # cut it off so batch[2] is only the xyz marker targets
        batch[1] = batch[1][:, 0:72]


        images_up_non_tensor = np.array(batch[0].numpy())



        INPUT_DICT['batch_images'] = np.copy(images_up_non_tensor)


        #here perform synthetic calibration noise over pmat and sobel filtered pmat.
        if CTRL_PNL['cal_noise'] == True:
            images_up_non_tensor = PreprocessingLib().preprocessing_add_calibration_noise(images_up_non_tensor,
                                                                                          pmat_chan_idx = (CTRL_PNL['num_input_channels_batch0']-2),
                                                                                          norm_std_coeffs = CTRL_PNL['norm_std_coeffs'],
                                                                                          is_training = is_training,
                                                                                          noise_amount = CTRL_PNL['cal_noise_amt'],
                                                                                          normalize_per_image = CTRL_PNL['normalize_per_image'])


        #print np.shape(images_up_non_tensor)

        if is_training == True: #only add noise to training images
            if CTRL_PNL['cal_noise'] == False:
                images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(images_up_non_tensor),
                                                                                    pmat_chan_idx = (CTRL_PNL['num_input_channels_batch0']-2),
                                                                                    norm_std_coeffs = CTRL_PNL['norm_std_coeffs'])
            else:
                images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(images_up_non_tensor),
                                                                                    pmat_chan_idx = (CTRL_PNL['num_input_channels_batch0']-1),
                                                                                    norm_std_coeffs = CTRL_PNL['norm_std_coeffs'])

        images_up_non_tensor = PreprocessingLib().preprocessing_pressure_map_upsample(images_up_non_tensor, multiple=2)

        images_up = Variable(torch.Tensor(images_up_non_tensor).type(CTRL_PNL['dtype']), requires_grad=False)


        if CTRL_PNL['incl_ht_wt_channels'] == True: #make images full of stuff
            weight_input = torch.ones((images_up.size()[0], images_up.size()[2] * images_up.size()[3])).type(CTRL_PNL['dtype'])
            weight_input *= batch[7].type(CTRL_PNL['dtype'])
            weight_input = weight_input.view((images_up.size()[0], 1, images_up.size()[2], images_up.size()[3]))
            height_input = torch.ones((images_up.size()[0], images_up.size()[2] * images_up.size()[3])).type(CTRL_PNL['dtype'])
            height_input *= batch[8].type(CTRL_PNL['dtype'])
            height_input = height_input.view((images_up.size()[0], 1, images_up.size()[2], images_up.size()[3]))
            images_up = torch.cat((images_up, weight_input, height_input), 1)


        targets, betas = Variable(batch[1].type(CTRL_PNL['dtype']), requires_grad=False), \
                         Variable(batch[2].type(CTRL_PNL['dtype']), requires_grad=False)

        angles_gt = Variable(batch[3].type(CTRL_PNL['dtype']), requires_grad=is_training)
        root_shift = Variable(batch[4].type(CTRL_PNL['dtype']), requires_grad=is_training)
        gender_switch = Variable(batch[5].type(CTRL_PNL['dtype']), requires_grad=is_training)
        synth_real_switch = Variable(batch[6].type(CTRL_PNL['dtype']), requires_grad=is_training)

        OUTPUT_EST_DICT = {}
        if CTRL_PNL['adjust_ang_from_est'] == True:
            OUTPUT_EST_DICT['betas'] = Variable(batch[9].type(CTRL_PNL['dtype']), requires_grad=is_training)
            OUTPUT_EST_DICT['angles'] = Variable(extra_smpl_angles.type(CTRL_PNL['dtype']), requires_grad=is_training)
            OUTPUT_EST_DICT['root_shift'] = Variable(extra_targets.type(CTRL_PNL['dtype']), requires_grad=is_training)
            if CTRL_PNL['full_body_rot'] == True:
                OUTPUT_EST_DICT['root_atan2'] = Variable(batch[12].type(CTRL_PNL['dtype']), requires_grad=is_training)

        if CTRL_PNL['depth_map_labels'] == True:
            if CTRL_PNL['depth_map_labels_test'] == True or is_training == True:
                INPUT_DICT['batch_mdm'] = batch[9+adj_ext_idx].type(CTRL_PNL['dtype'])
                INPUT_DICT['batch_cm'] = batch[10+adj_ext_idx].type(CTRL_PNL['dtype'])
        else:
            INPUT_DICT['batch_mdm'] = None
            INPUT_DICT['batch_cm'] = None


        #print images_up.size(), CTRL_PNL['num_input_channels_batch0']

        if CTRL_PNL['omit_cntct_sobel'] == True:
            images_up[:, 0, :, :] *= 0

            if CTRL_PNL['cal_noise'] == True:
                images_up[:, CTRL_PNL['num_input_channels_batch0'], :, :] *= 0
            else:
                images_up[:, CTRL_PNL['num_input_channels_batch0']-1, :, :] *= 0



        if CTRL_PNL['use_hover'] == False and CTRL_PNL['adjust_ang_from_est'] == True:
            images_up[:, 1, :, :] *= 0


        scores, OUTPUT_DICT = model.forward_kinematic_angles(images=images_up,
                                                             gender_switch=gender_switch,
                                                             synth_real_switch=synth_real_switch,
                                                             CTRL_PNL=CTRL_PNL,
                                                             OUTPUT_EST_DICT=OUTPUT_EST_DICT,
                                                             targets=targets,
                                                             is_training=is_training,
                                                             betas=betas,
                                                             angles_gt=angles_gt,
                                                             root_shift=root_shift,
                                                             )  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.


        INPUT_DICT['batch_images'] = images_up.data
        INPUT_DICT['batch_targets'] = targets.data

        for i in range(INPUT_DICT['batch_images'].size()[1]):
            print 'max: ', torch.max(INPUT_DICT['batch_images'][0, i, :, :]).cpu().data.numpy(),
        for i in range(INPUT_DICT['batch_images'].size()[1]):
            print '  sum: ', torch.sum(INPUT_DICT['batch_images'][0, i, :, :]).cpu().data.numpy(),
        print

        return scores, INPUT_DICT, OUTPUT_DICT

