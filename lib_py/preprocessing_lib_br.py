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

from scipy.ndimage.filters import gaussian_filter


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
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)



class PreprocessingLib():
    def __init__(self):
        pass


    def preprocessing_add_image_noise(self, images, pmat_chan_idx, norm_std_coeffs):

        queue = np.copy(images[:, pmat_chan_idx:pmat_chan_idx+2, :, :])
        queue[queue != 0] = 1.


        x = np.arange(-10, 10)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=1) - ss.norm.cdf(xL,scale=1)  # scale is the standard deviation using a cumulative density function
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        image_noise = np.random.choice(x, size=(images.shape[0], 2, images.shape[2], images.shape[3]), p=prob)

        
        image_noise = image_noise*queue
        image_noise = image_noise.astype(float)
        #image_noise[:, 0, :, :] /= 11.70153502792190
        #image_noise[:, 1, :, :] /= 45.61635847182483
        image_noise[:, 0, :, :] *= norm_std_coeffs[4]
        image_noise[:, 1, :, :] *= norm_std_coeffs[5]

        images[:, pmat_chan_idx:pmat_chan_idx+2, :, :] += image_noise

        #print images[0, 0, 50, 10:25], 'added noise'

        #clip noise so we dont go outside sensor limits
        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100/11.70153502792190)
        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100.*norm_std_coeffs[4])
        images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 10000.)
        #images[:, pmat_chan_idx+1, :, :] = np.clip(images[:, pmat_chan_idx+1, :, :], 0, 10000)
        return images


    def preprocessing_add_calibration_noise(self, images, pmat_chan_idx, norm_std_coeffs, is_training, noise_amount, normalize_per_image):

        if is_training == True:
            variation_amount = float(noise_amount)
            print "ADDING CALIB NOISE", variation_amount

            #pmat_contact_orig = np.copy(images[:, pmat_chan_idx, :, :])
            #pmat_contact_orig[pmat_contact_orig != 0] = 1.
            #sobel_contact_orig = np.copy(images[:, pmat_chan_idx+1, :, :])
            #sobel_contact_orig[sobel_contact_orig != 0] = 1.

            for map_index in range(images.shape[0]):

                pmat_contact_orig = np.copy(images[map_index, pmat_chan_idx, :, :])
                pmat_contact_orig[pmat_contact_orig != 0] = 1.
                sobel_contact_orig = np.copy(images[map_index, pmat_chan_idx + 1, :, :])
                sobel_contact_orig[sobel_contact_orig != 0] = 1.


                # first multiply
                amount_to_mult_im = random.normalvariate(mu = 1.0, sigma = variation_amount) #mult a variation of 10%
                amount_to_mult_sobel = random.normalvariate(mu = 1.0, sigma = variation_amount) #mult a variation of 10%
                images[map_index, pmat_chan_idx, :, :] = images[map_index, pmat_chan_idx, :, :] * amount_to_mult_im
                images[map_index, pmat_chan_idx+1, :, :] = images[map_index, pmat_chan_idx+1, :, :] * amount_to_mult_sobel

                # then add
                #amount_to_add_im = random.normalvariate(mu = 0.0, sigma = (1./11.70153502792190)*(98.666 - 0.0)*0.1) #add a variation of 10% of the range
                #amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = (1./45.61635847182483)*(386.509 - 0.0)*0.1) #add a variation of 10% of the range

                if normalize_per_image == True:
                    amount_to_add_im = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[4]*(70. - 0.0)*variation_amount) #add a variation of 10% of the range
                    amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[5]*(70. - 0.0)*variation_amount) #add a variation of 10% of the range
                else:
                    amount_to_add_im = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[4]*(98.666 - 0.0)*variation_amount) #add a variation of 10% of the range
                    amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[5]*(386.509 - 0.0)*variation_amount) #add a variation of 10% of the range

                images[map_index, pmat_chan_idx, :, :] = images[map_index, pmat_chan_idx, :, :] + amount_to_add_im
                images[map_index, pmat_chan_idx+1, :, :] = images[map_index, pmat_chan_idx+1, :, :] + amount_to_add_sobel
                images[map_index, pmat_chan_idx, :, :] = np.clip(images[map_index, pmat_chan_idx, :, :], a_min = 0., a_max = 10000)
                images[map_index, pmat_chan_idx+1, :, :] = np.clip(images[map_index, pmat_chan_idx+1, :, :], a_min = 0., a_max = 10000)

                #cut out the background. need to do this after adding.
                images[map_index, pmat_chan_idx, :, :] *= pmat_contact_orig#[map_index, :, :]
                images[map_index, pmat_chan_idx+1, :, :] *= sobel_contact_orig#[map_index, :, :]


                amount_to_gauss_filter_im = random.normalvariate(mu = 0.5, sigma = variation_amount)
                amount_to_gauss_filter_sobel = random.normalvariate(mu = 0.5, sigma = variation_amount)
                images[map_index, pmat_chan_idx, :, :] = gaussian_filter(images[map_index, pmat_chan_idx, :, :], sigma= amount_to_gauss_filter_im) #pmap
                images[map_index, pmat_chan_idx+1, :, :] = gaussian_filter(images[map_index, pmat_chan_idx+1, :, :], sigma= amount_to_gauss_filter_sobel) #sobel #NOW


        else:  #if its NOT training we should still blur things by 0.5
            for map_index in range(images.shape[0]):
               # print pmat_chan_idx, images.shape, 'SHAPE'
                images[map_index, pmat_chan_idx, :, :] = gaussian_filter(images[map_index, pmat_chan_idx, :, :], sigma= 0.5) #pmap
                images[map_index, pmat_chan_idx+1, :, :] = gaussian_filter(images[map_index, pmat_chan_idx+1, :, :], sigma= 0.5) #sobel

        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100/11.70153502792190)
            if normalize_per_image == False:
                images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100.*norm_std_coeffs[4])
            else:
                images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 10000.)



        #now calculate the contact map AFTER we've blurred it
        pmat_contact = np.copy(images[:, pmat_chan_idx:pmat_chan_idx+1, :, :])
        #pmat_contact[pmat_contact != 0] = 100./41.80684362163343
        pmat_contact[pmat_contact != 0] = 100.*norm_std_coeffs[0]
        images = np.concatenate((pmat_contact, images), axis = 1)

        #for i in range(0, 20):
        #    VisualizationLib().visualize_pressure_map(images[i, 0, :, :] * 20., None, None,
        #                                              images[i, 1, :, :] * 20., None, None,
        #                                              block=False)
        #    time.sleep(0.5)



        return images


    def preprocessing_blur_images(self, x_data, mat_size, sigma):

        x_data_return = []
        for map_index in range(len(x_data)):
            p_map = np.reshape(x_data[map_index], mat_size)

            p_map = gaussian_filter(p_map, sigma= sigma)

            x_data_return.append(p_map.flatten())

        return x_data_return




    def preprocessing_create_pressure_angle_stack(self,x_data, mat_size, CTRL_PNL):
        '''This is for creating a 2-channel input using the height of the bed. '''

        if CTRL_PNL['verbose']: print np.max(x_data)
        x_data = np.clip(x_data, 0, 100)

        print "normalizing per image", CTRL_PNL['normalize_per_image']

        p_map_dataset = []
        for map_index in range(len(x_data)):
            # print map_index, self.mat_size, 'mapidx'
            # Resize mat to make into a matrix

            p_map = np.reshape(x_data[map_index], mat_size)

            if CTRL_PNL['normalize_per_image'] == True:
                p_map = p_map * (20000./np.sum(p_map))

            if mat_size == (84, 47):
                p_map = p_map[10:74, 10:37]

            # this makes a sobel edge on the image
            sx = ndimage.sobel(p_map, axis=0, mode='constant')
            sy = ndimage.sobel(p_map, axis=1, mode='constant')
            p_map_inter = np.hypot(sx, sy)
            if CTRL_PNL['clip_sobel'] == True:
                p_map_inter = np.clip(p_map_inter, a_min=0, a_max = 100)

            if CTRL_PNL['normalize_per_image'] == True:
                p_map_inter = p_map_inter * (20000. / np.sum(p_map_inter))

            #print np.sum(p_map), 'sum after norm'
            p_map_dataset.append([p_map, p_map_inter])

        return p_map_dataset


    def preprocessing_pressure_map_upsample(self, data, multiple, order=1):
        '''Will upsample an incoming pressure map dataset'''
        p_map_highres_dataset = []


        if len(np.shape(data)) == 3:
            for map_index in range(len(data)):
                #Upsample the current map using bilinear interpolation
                p_map_highres_dataset.append(
                        ndimage.zoom(data[map_index], multiple, order=order))
        elif len(np.shape(data)) == 4:
            for map_index in range(len(data)):
                p_map_highres_dataset_subindex = []
                for map_subindex in range(len(data[map_index])):
                    #Upsample the current map using bilinear interpolation
                    p_map_highres_dataset_subindex.append(ndimage.zoom(data[map_index][map_subindex], multiple, order=order))
                p_map_highres_dataset.append(p_map_highres_dataset_subindex)

        return p_map_highres_dataset



    def pad_pressure_mats(self,NxHxWimages):
        padded = np.zeros((NxHxWimages.shape[0],NxHxWimages.shape[1]+20,NxHxWimages.shape[2]+20))
        padded[:,10:74,10:37] = NxHxWimages
        NxHxWimages = padded
        return NxHxWimages


    def preprocessing_per_im_norm(self, images, CTRL_PNL):

        if CTRL_PNL['depth_map_input_est'] == True:
            pmat_sum = 1./(torch.sum(torch.sum(images[:, 4, :, :], dim=1), dim=1)/100000.)
            sobel_sum = 1./(torch.sum(torch.sum(images[:, 5, :, :], dim=1), dim=1)/100000.)

            print "ConvNet input size: ", images.size(), pmat_sum.size()
            for i in range(images.size()[1]):
                print i, torch.min(images[0, i, :, :]), torch.max(images[0, i, :, :])

            images[:, 4, :, :] = (images[:, 4, :, :].permute(1, 2, 0)*pmat_sum).permute(2, 0, 1)
            images[:, 5, :, :] = (images[:, 5, :, :].permute(1, 2, 0)*sobel_sum).permute(2, 0, 1)

        else:
            pmat_sum = 1./(torch.sum(torch.sum(images[:, 1, :, :], dim=1), dim=1)/100000.)
            sobel_sum = 1./(torch.sum(torch.sum(images[:, 2, :, :], dim=1), dim=1)/100000.)

            print "ConvNet input size: ", images.size(), pmat_sum.size()
            for i in range(images.size()[1]):
                print i, torch.min(images[0, i, :, :]), torch.max(images[0, i, :, :])


            images[:, 1, :, :] = (images[:, 1, :, :].permute(1, 2, 0)*pmat_sum).permute(2, 0, 1)
            images[:, 2, :, :] = (images[:, 2, :, :].permute(1, 2, 0)*sobel_sum).permute(2, 0, 1)



        #do this ONLY to pressure and sobel. scale the others to get them in a reasonable range, by a constant factor.


        return images